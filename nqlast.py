"""Implements an EDSL for constructing Turing machines without subclassing
AstMachine."""

from collections import namedtuple
from framework import Halt, InsnInfo, Machine, Goto, Label, Register, State, Subroutine, cfg_optimizer, make_bits, make_dispatcher, memo
from astt import *

class SubEmitter:
    """Tracks state while lowering a _SubDef to a call sequence."""

    def __init__(self, register_map, machine_builder, name):
        self._register_map = register_map
        self._machine_builder = machine_builder
        self._scratch_next = 0
        self._scratch_used = []
        self._scratch_free = []
        self._output = []
        self._return_label = None
        self.break_label = None
        self.name = name

    def emit_transfer(self, *regs):
        self._output.append(self._machine_builder.transfer(*regs))

    def emit_halt(self):
        self._output.append(self._machine_builder.halt())

    def emit_noop(self):
        self._output.append(self._machine_builder.noop(0))

    def emit_label(self, label):
        self._output.append(Label(label))

    def emit_goto(self, label):
        self._output.append(Goto(label))

    def emit_return(self):
        if self.name == 'main':
            self.emit_halt()
            return
        if not self._return_label:
            self._return_label = self.gensym()
        self.emit_goto(self._return_label)

    def close_return(self):
        if self._return_label:
            self.emit_label(self._return_label)

    def emit_inc(self, reg):
        self._output.append(reg.inc)

    def emit_dec(self, reg):
        self._output.append(reg.dec)

    def emit_call(self, func_name, args):
        assert len(self._scratch_used) == 0
        if func_name.startswith('noop_'):
            self._output.append(self._machine_builder.noop(int(func_name[5:])))
        elif func_name.startswith('builtin_'):
            getattr(self, 'emit_' + func_name)(*args)
        else:
            func = self._machine_builder.instantiate(func_name, tuple(arg.name for arg in args))
            self._output.append(func)

    def emit_builtin_pair(self, out, in1, in2):
        t0 = self.get_temp()
        extract = self.gensym()
        nextdiag = self.gensym()
        done = self.gensym()
        self.emit_label(extract)
        self.emit_dec(in1)
        self.emit_goto(nextdiag)
        self.emit_inc(t0)
        self.emit_inc(in2)
        self.emit_goto(extract)
        self.emit_label(nextdiag)
        self.emit_dec(in2)
        self.emit_goto(done)
        self.emit_inc(t0)
        self.emit_transfer(in2, in1)
        self.emit_goto(extract)
        self.emit_label(done)
        self.emit_transfer(out)
        self.emit_transfer(t0, out)
        self.put_temp(t0)

    def emit_builtin_unpair(self, out1, out2, in1):
        t0 = self.get_temp()
        self.emit_transfer(in1, t0)
        self.emit_transfer(out1)
        self.emit_transfer(out2)

        nextdiag = self.gensym()
        nextstep = self.gensym()
        done = self.gensym()

        self.emit_label(nextstep)
        self.emit_dec(t0)
        self.emit_goto(done)
        self.emit_inc(out1)
        self.emit_dec(out2)
        self.emit_goto(nextdiag)
        self.emit_goto(nextstep)
        self.emit_label(nextdiag)
        self.emit_transfer(out1, out2)
        self.emit_goto(nextstep)
        self.emit_label(done)

        self.put_temp(t0)

    def emit_builtin_move(self, to_, from_):
        t0 = self.get_temp()
        self.emit_transfer(from_, t0)
        self.emit_transfer(to_)
        self.emit_transfer(t0, to_)
        self.put_temp(t0)

    def resolve(self, regname):
        reg = self._register_map.get(regname) or '_G' + regname
        return self._machine_builder.register(reg) if isinstance(reg,str) else reg

    def put_temp(self, reg):
        self._scratch_used.remove(reg)
        self._scratch_free.append(reg)

    def get_temp(self):
        if self._scratch_free:
            var = self._scratch_free.pop()
        else:
            self._scratch_next += 1
            var = self._machine_builder.register('_scratch_' + str(self._scratch_next))
        self._scratch_used.append(var)
        return var

    def gensym(self):
        self._machine_builder._gensym += 1
        return 'gen' + str(self._machine_builder._gensym)

class AstMachine:
    """Subclassable class of utilities for constructing Turing machines using
    BDD-compressed register machines."""
    pc_bits = 0
    # Quick=0: Print TM
    # Quick=1: Simulate TM, print all steps
    # Quick=2: Simulate TM, print at dispatch
    # Quick=3: Simulate compressed register machine
    # Quick=4: as Quick=3 except subroutines can cheat
    # Quick=5: subroutines can cheat to the extent of storing non-integers

    def __init__(self, ast, control_args):
        self._ast = ast
        self._fun_instances = {}
        self._gensym = 0
        self._nextreg = 0
        self._memos = {}
        self.control_args = control_args

    @memo
    def instantiate(self, name, args):
        defn = self._ast.by_name[name]
        assert isinstance(defn, ProcDef)
        emit = SubEmitter(dict(zip(defn.parameters, args)), self, name)
        defn.children[0].emit_stmt(emit)
        if name != 'main':
            emit.close_return()
        return self.makesub(name=name + '(' + ','.join(args) + ')', *emit._output)

    def main(self):
        return self.instantiate('main', ())

    # leaf procs which implement register machine operations
    # on entry to a leaf proc the tape head is just after the PC

    @memo
    def reg_incr(self, index):
        """Primitive subroutine which decrements a register."""
        if index == -2:
            entry = self.register_common().inc
        else:
            entry = State()
            entry.be(move=1, next1=entry, next0=self.reg_incr(index-1), name='reg_incr.'+str(index))

        return entry

    @memo
    def reg_decr(self, index):
        """Primitive subroutine which decrements a register.  The PC will be
        incremented by 2 if successful; if the register was zero, it will be
        unchanged and the PC will be incremented by 1."""
        if index == -2:
            entry = self.register_common().dec
        else:
            entry = State()
            entry.be(move=1, next1=entry, next0=self.reg_decr(index-1), name='reg_decr.'+str(index))

        return entry

    @memo
    def reg_init(self):
        """Primitive subroutine which initializes a register.  Call this N
        times before using registers less than N."""
        return Subroutine(self.register_common().init, 0, 'reg_init')

    @memo
    def register_common(self):
        """Primitive register operations start with the tape head on the first
        1 bit of a register, and exit by running back into the dispatcher."""
        (inc_shift_1, inc_shift_0, dec_init, dec_check, dec_scan_1,
         dec_scan_0, dec_scan_done, dec_shift_0, dec_shift_1, dec_restore,
         return_0, return2_0, return_1, return2_1, init_f1, init_f2,
         init_scan_1, init_scan_0) = (State() for i in range(18))

        # Initialize routine
        init_f1.be(move=1, next=init_f2, name='init.f1')
        init_f2.be(move=1, next=init_scan_0, name='init.f2')
        init_scan_1.be(move=1, next1=init_scan_1, next0=init_scan_0, name='init.scan_1') # only 0 is possible
        init_scan_0.be(write0='1', move0=-1, next0=return_1, move1=1, next1=init_scan_1, name='init.scan_0')

        # Increment the register, the first 1 bit of which is under the tape head
        inc_shift_1.be(move=1, write='1', next0=inc_shift_0, next1=inc_shift_1, name='inc.shift_1')
        inc_shift_0.be(write='0', next0=return_0, move0=-1, next1=inc_shift_1, move1=1, name='inc.shift_0')

        # Decrementing is a bit more complicated, we need to mark the register we're on
        dec_init.be(write='0', move=1, next=dec_check, name='dec.init')
        dec_check.be(move0=-1, next0=dec_restore, move1=1, next1=dec_scan_1, name='dec.check')

        dec_scan_1.be(move=1, next1=dec_scan_1, next0=dec_scan_0, name='dec.scan_1')
        dec_scan_0.be(move1=1, next1=dec_scan_1, move0=-1, next0=dec_scan_done, name='dec.scan_0')
        # scan_done = on 0 after last reg
        dec_scan_done.be(move=-1, next=dec_shift_0, name='dec.scan_done')
        dec_shift_0.be(write='0', move0=-1, next0=return2_0, move1=-1, next1=dec_shift_1, name='dec.shift_0')
        # if shifting 0 onto 0, we're moving the marker we created
        # let it overlap the fence
        dec_shift_1.be(write='1', move=-1, next0=dec_shift_0, next1=dec_shift_1, name='dec.shift_1')

        dec_restore.be(write='1', move=-1, next=return_1, name='dec.restore')

        return_0.be(move=-1, next0=self.nextstate(), next1=return_1, name='return.0')
        return2_0.be(move=-1, next0=self.nextstate_2(), next1=return2_1, name='return2.0')
        return_1.be(move=-1, next0=return_0, next1=return_1, name='return.1')
        return2_1.be(move=-1, next0=return2_0, next1=return2_1, name='return2.1')

        return namedtuple('register_common', 'inc dec init')(inc_shift_1, dec_init, init_f1)

    # Implementing the subroutine model

    @memo
    def dispatchroot(self):
        """A Turing state which issues the correct operation starting from the first PC bit."""
        return State()

    @memo
    def nextstate(self):
        """A Turing state which increments PC by 1, with the tape head on the last PC bit."""
        return self.dispatch_order(0, 1)

    @memo
    def nextstate_2(self):
        """A Turing state which increments PC by 2, with the tape head on the last PC bit."""
        return State(move=-1, next=self.dispatch_order(1, 1), name='nextstate_2')

    @memo
    def dispatch_order(self, order, carry_bit):
        """Constructs Turing states which move from the work area back to the PC head.

        On entry, the head should be order bits left of the rightmost bit of the program
        counter; if carry_bit is set, the bit the head is on will be incremented."""
        if order == self.pc_bits:
            return State(move=+1, next=self.dispatchroot(), name='!ENTRY')
        assert order < self.pc_bits
        if carry_bit:
            return State(write0='1', next0=self.dispatch_order(order + 1, 0),
                         write1='0', next1=self.dispatch_order(order + 1, 1),
                         move=-1, name='dispatch.{}.carry'.format(order))
        else:
            return State(next=self.dispatch_order(order + 1, 0), move=-1,
                         name='dispatch.{}'.format(order))

    @memo
    def noop(self, order):
        """A subprogram of given size which does nothing.

        Used automatically to maintain alignment."""
        reverse = State(move=-1, next=self.dispatch_order(order, 1), name='noop.{}'.format(order))
        return Subroutine(reverse, order, reverse.name)

    @memo
    def halt(self):
        """A subprogram which halts the Turing machine when your work is done."""
        return Subroutine(Halt(), 0, 'halt')

    @memo
    def jump(self, order, rel_pc, sub_name):
        """A subprogram which replaces a suffix of the PC, for relative jumps.

        Used automatically by the Goto operator."""
        assert rel_pc < (1 << (order + 1))
        steps = [State() for i in range(order + 2)]
        steps[order+1] = self.dispatch_order(order, rel_pc >> order)
        steps[0].be(move=-1, next=steps[1], \
            name='{}.jump({},{},{})'.format(sub_name, rel_pc, order, 0))
        for i in range(order):
            bit = str((rel_pc >> i) & 1)
            steps[i+1].be(move=-1, next=steps[i+2], write=bit, \
                name='{}.jump({},{},{})'.format(sub_name, rel_pc, order, i+1))

        return Subroutine(steps[0], 0, '{}.jump({},{})'.format(sub_name, rel_pc, order))

    @memo
    def rjump(self, rel_pc):
        """A subprogram which adds a constant to the PC, for relative jumps."""
        steps = [(State(), State()) for i in range(self.pc_bits + 1)]
        steps.append(2 * (self.dispatch_order(self.pc_bits, 0),))
        steps[0][0].be(move=-1, next=steps[1][0], name='rjump({})({})'.format(rel_pc, 0))
        for i in range(self.pc_bits):
            bit = (rel_pc >> i) & 1
            steps[i+1][0].be(move=-1, next0=steps[i+2][0], write0=str(bit), \
                next1=steps[i+2][bit], write1=str(1-bit), \
                name='rjump({})({})'.format(rel_pc, i+1))
            steps[i+1][1].be(move=-1, next0=steps[i+2][bit], write0=str(1-bit), \
                next1=steps[i+2][1], write1=str(bit), \
                name='rjump({})({}+)'.format(rel_pc, i+1))

        return Subroutine(steps[0][0], 0, 'rjump({})'.format(rel_pc))

    # TODO: subprogram compilation needs to be substantially lazier in order to do
    # effective inlining and register allocation
    def makesub(self, *parts, name):
        """Assigns PC values within a subprogram and creates the dispatcher."""
        # first find out where everything is and how big I am

        label_offsets = {}
        label_map = {}
        goto_map = {}
        real_parts = []
        offset = 0

        if not self.control_args.no_cfg_optimize:
            parts = cfg_optimizer(parts)

        if name == 'main()':
            # inject code to initialize registers (a bit of a hack)
            regcount = self._nextreg
            while regcount & (regcount - 1):
                regcount += 1
            parts = regcount * (self.reg_init(), ) + parts

        for part in parts:
            if isinstance(part, Label):
                # labels take up no space
                label_offsets[part.name] = offset
                label_map.setdefault(offset, []).append(part.name)
                continue # not a real_part

            if isinstance(part, Goto):
                goto_map[offset] = part.name

            # parts must be aligned
            while offset % part.size:
                noop_order = (offset & -offset).bit_length() - 1
                offset += 1 << noop_order
                real_parts.append(self.noop(noop_order))

            real_parts.append(part)
            offset += part.size

        assert offset > 0

        order = 0
        while offset > (1 << order):
            order += 1

        while offset < (1 << order):
            noop_order = (offset & -offset).bit_length() - 1
            offset += 1 << noop_order
            real_parts.append(self.noop(noop_order))

        offset = 0
        child_map = {}

        jumps_required = set()

        for part in real_parts:
            if isinstance(part, Goto):
                jump_order = 0
                target = label_offsets[part.name]
                while True:
                    base = (offset >> jump_order) << jump_order
                    rel = target - base
                    if rel >= 0 and rel < (1 << (jump_order + 1)):
                        jumps_required.add((jump_order, rel))
                        break
                    jump_order += 1
            offset += part.size
        offset = 0

        for part in real_parts:
            if isinstance(part, Goto):
                assert part.name in label_offsets
                target = label_offsets[part.name]
                if self.control_args.relative_jumps:
                    part = self.rjump(target - offset)
                else:
                    part = None
                    for jump_order in range(order + 1):
                        base = (offset >> jump_order) << jump_order
                        rel = target - base
                        if (jump_order, rel) in jumps_required:
                            part = self.jump(jump_order, rel, name)
                            # don't break, we want to take the largest reqd jump
                            # except for very short jumps, those have low enough
                            # entropy to be worthwhile
                            if jump_order < 3:
                                break
                    assert part
            offset_bits = make_bits(offset >> part.order, order - part.order)
            goto_line = goto_map.get(offset)
            label_line = label_map.get(offset)
            child_map[offset_bits] = InsnInfo(part, label_line, goto_line)
            offset += 1 << part.order

        return Subroutine(make_dispatcher(child_map, name, order), order, name, child_map=child_map)

    # Utilities...
    @memo
    def register(self, name):
        """Assigns a name to a register, and creates the primitive inc/dec routines."""
        index = self._nextreg
        self._nextreg += 1

        inc = Subroutine(self.reg_incr(index), 0, 'reg_incr('+name+')')
        dec = Subroutine(self.reg_decr(index), 0, 'reg_decr('+name+')', is_decrement=True)

        return Register(name, index, inc, dec)

    @memo
    def transfer(self, source, *to):
        """Subprogram which moves values between registers.

        The source register will be cleared, and its value will be added to each to register."""
        name = 'transfer(' + ','.join([source.name] + [x.name for x in sorted(to)]) + ')'
        return self.makesub(
            Label('again'),
            source.dec,
            Goto('zero'),
            *([tox.inc for tox in sorted(to)] + [
                Goto('again'),
                Label('zero'),
            ]),
            name=name
        )

def harness(ast, args):
    mach1 = AstMachine(ast, args)
    mach1.pc_bits = 50
    order = mach1.main().order
    mach2 = AstMachine(ast, args)
    mach2.pc_bits = order
    Machine(mach2).harness(args)
