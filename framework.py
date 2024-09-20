"""Framework for building Turing machines using a register machine abstraction
and binary decision diagrams in place of subprograms."""
# Tape layout: PC:bit[NNN] 0 0 ( 1 1* 0 )*
#
# each thing after the PC is a unary register.
#
# There is a "dispatch" state which assumes the head is at position zero, and
# reads PC bits through a decision tree to find out what to do.
#
# The decision tree has shared subtrees - this is how we handle "subroutines".
# Naturally these shared subtrees have to handle different "contexts".
#
# we shift 1 left of the PC MSB during carry phases; the initial state is the
# leftmost shift state, so the total shift is always non-negative.

from collections import namedtuple
import argparse

class Halt:
    """Special machine state which halts the Turing machine."""
    def __init__(self):
        self.name = 'HALT'

class State:
    """Represents a Turing machine state.

    Instances of State can be initialized either at construction or using
    the be() method; the latter allows for cyclic graphs to be defined."""
    def __init__(self, **kwargs):
        self.set = False
        self.name = '**UNINITIALIZED**'
        if kwargs:
            self.be(**kwargs)

    def be(self, name, move=None, next=None, write=None,
           move0=None, next0=None, write0=None,
           move1=None, next1=None, write1=None):
        """Defines a Turing machine state.

        The movement direction, next state, and new tape value can be defined
        depending on the old tape value, or for both tape values at the same time.
        Next state and direction must be provided, tape value can be omitted for no change."""
        assert not self.set
        self.set = True
        self.name = name
        self.move0 = move0 or move
        self.move1 = move1 or move
        self.next0 = next0 or next
        self.next1 = next1 or next
        self.write0 = write0 or write or '0'
        self.write1 = write1 or write or '1'
        assert self.move0 in (-1, 1)
        assert self.move1 in (-1, 1)
        assert self.write0 in ('0', '1')
        assert self.write1 in ('0', '1')
        assert isinstance(self.name, str)
        assert isinstance(self.next0, State) or isinstance(self.next0, Halt)
        assert isinstance(self.next1, State) or isinstance(self.next1, Halt)

    def clone(self, other):
        """Makes this state equivalent to another state, which must already be initialized."""
        assert isinstance(other, State) and other.set
        self.be(name=other.name, move0=other.move0, next0=other.next0,
                write0=other.write0, move1=other.move1, next1=other.next1,
                write1=other.write1)

def make_bits(num, bits):
    """Constructs a bit string of length=bits for an integer num."""
    assert num < (1 << bits)
    if bits == 0:
        return ''
    return '{num:0{bits}b}'.format(num=num, bits=bits)

def memo(func):
    """Decorator which memoizes a method, so it will be called once with a
    given set of arguments."""
    def _wrapper(self, *args):
        key = (func,) + args
        if key not in self._memos:
            self._memos[key] = None
            self._memos[key] = func(self, *args)
        if not self._memos[key]:
            print("recursion detected", func.__name__, repr(args))
            assert False
        return self._memos[key]

    return _wrapper

Label = namedtuple('Label', ['name'])
Label.size = 0
Label.is_decrement = False
Goto = namedtuple('Goto', ['name'])
Goto.size = 1
Goto.is_decrement = False
Register = namedtuple('Register', 'name index inc dec')

class Subroutine:
    """Class wrapping a compiled subprogram, which is an internal node in the
    program BDD.

    A subprogram consumes a power-of-two number of PC values, and can appear
    at any correctly aligned PC; the entry state is entered with the tape head
    on the first bit of the subprogram's owned portion of the PC."""
    def __init__(self, entry, order, name, child_map=None, is_decrement=False):
        self.entry = entry
        self.name = name
        self.order = order
        self.size = 1 << order
        self.is_decrement = is_decrement
        self.child_map = child_map or {}

InsnInfo = namedtuple('InsnInfo', 'sub labels goto')

def make_dispatcher(child_map, name, order, at_prefix=''):
    """Constructs one or more dispatch states to route to a child map.

    Each key in the child map must be a binary string no longer than
    the order, and every binary string of length equal to the order must
    have exactly one child map key as a prefix.  The generated states will
    read bits going right and fall into the child states after reading
    exactly the prefix."""
    if at_prefix in child_map:
        return child_map[at_prefix].sub.entry
    assert len(at_prefix) <= order
    switch = State()
    switch.be(move=1, name=name + '[' + at_prefix + ']',
              next0=make_dispatcher(child_map, name, order, at_prefix + '0'),
              next1=make_dispatcher(child_map, name, order, at_prefix + '1'))
    return switch

def cfg_optimizer(parts):
    parts = list(parts)

    # Thread jumps to jumps
    # Delete jumps to the next instruction
    counter = 0
    label_map = {}
    rlabel_map = {}
    goto_map = {}
    labels = []
    for insn in parts:
        if isinstance(insn, Label):
            labels.append(insn.name)
        else:
            for label in labels:
                label_map[label] = counter
                rlabel_map[counter] = label
            labels = []
            if isinstance(insn, Goto):
                goto_map[counter] = insn.name
            counter += 1
    for label in labels:
        label_map[label] = counter
        rlabel_map[counter] = label

    def follow(count):
        for _ in range(10):
            if count not in goto_map:
                break
            count = label_map[goto_map[count]]
        return count
    # print(repr(parts))

    counter = 0
    for index, insn in enumerate(parts):
        if isinstance(insn, Label):
            continue
        if isinstance(insn, Goto):
            direct_goes_to = label_map[goto_map[counter]]
            goes_to = follow(direct_goes_to)
            next_goes_to = goto_map.get(counter+1) and follow(counter+1)

            # print("CFGO", insn.name, counter, goes_to, next_goes_to)
            if goes_to == counter + 1 or goes_to == next_goes_to:
                parts[index] = None
            elif direct_goes_to != goes_to:
                parts[index] = Goto(rlabel_map[goes_to])
        counter += 1

    # print(repr(parts))

    # Delete dead code

    # label_to_index = {}
    # for index, insn in enumerate(parts):
    #     if isinstance(insn, Label):
    #         label_to_index[insn.name] = index

    # grey_index = [0]
    # black_index = set()
    # while grey_index:
    #     ix = grey_index.pop()
    #     if ix in black_index or ix >= len(parts):
    #         continue
    #     black_index.add(ix)

    #     if isinstance(insn, Goto):
    #         grey_index.append(label_to_index[insn.name])
    #     else:
    #         grey_index.append(ix + 1)
    #         if insn and insn.is_decrement:
    #             grey_index.append(ix + 2)

    # for index in range(len(parts)):
    #     if index not in black_index:
    #         print("DEAD CODE")
    #         parts[index] = None

    return tuple(p for p in parts if p)

class Machine:
    """Manipulates and debugs the generated Turing machine for a AstMachine."""
    def __init__(self, builder):
        self.main = builder.main()

        if self.main.order != builder.pc_bits:
            print('pc_bits does not match calculated main order:', self.main.order, builder.pc_bits)
            assert False

        builder.dispatchroot().clone(self.main.entry)
        self.entry = builder.dispatch_order(builder.pc_bits, 0)

        self.state = self.entry
        self.left_tape = []
        self.current_tape = '0'
        self.right_tape = []
        self.longest_label = max(len(state.name) for state in self.reachable())

    def harness(self, args):
        """Processes command line arguments and runs the test harness for a machine."""

        if not args.dont_compress:
            self.compress()

        if args.print_subs:
            self.print_subs()

        if args.print_tm:
            self.print_machine()

        if args.run_tm:
            while isinstance(self.state, State):
                self.tm_step()

    def compress(self):
        """Combine pairs of equivalent states in the turing machine."""
        while True:
            did_work = False
            unique_map = {}
            replacement_map = {}

            for state in self.reachable():
                tup = (state.next0, state.next1, state.write0, state.write1,
                       state.move0, state.move1)
                if tup in unique_map:
                    replacement_map[state] = unique_map[tup]
                else:
                    unique_map[tup] = state

            for state in self.reachable():
                if state.next0 in replacement_map:
                    did_work = True
                    state.next0 = replacement_map[state.next0]
                if state.next1 in replacement_map:
                    did_work = True
                    state.next1 = replacement_map[state.next1]

            if self.entry in replacement_map:
                did_work = True
                self.entry = replacement_map[self.entry]

            if not did_work:
                break

    def print_subs(self):
        """Dump the subroutines used by this machine."""

        stack = [self.main]
        seen = set()
        while stack:
            subp = stack.pop()
            if subp in seen:
                continue
            seen.add(subp)
            print()
            print('NAME:', subp.name, 'ORDER:', subp.order)
            for offset, entry in sorted(subp.child_map.items()):
                while len(offset) < subp.order:
                    offset = offset + ' '
                display = '    {offset} -> {child}'.format(offset=offset, child=entry.sub.name)
                if entry.goto:
                    display += ' -> ' + entry.goto
                for label in entry.labels or ():
                    display += ' #' + label
                print(display)
                stack.append(entry.sub)

    def reachable(self):
        """Enumerates reachable states for the generated Turing machine."""
        queue = [self.entry]
        seen = []
        seen_set = set()
        while queue:
            state = queue.pop()
            if isinstance(state, Halt) or state in seen_set:
                continue
            if not state.set:
                continue
            seen_set.add(state)
            seen.append(state)
            queue.append(state.next1)
            queue.append(state.next0)
        return seen

    def print_machine(self):
        """Prints the state-transition table for the generated Turing machine."""
        reachable = sorted(self.reachable(), key=lambda x: x.name)

        count = {}
        for state in reachable:
            count[state.name] = count.get(state.name, 0) + 1

        index = {}
        renumber = {}
        for state in reachable:
            if count[state.name] == 1:
                continue
            index[state.name] = index.get(state.name, 0) + 1
            renumber[state] = state.name + '(#' + str(index[state.name]) + ')'

        dirmap = {1: 'R', -1: 'L'}
        for state in sorted(self.reachable(), key=lambda x: x.name):
            print(renumber.get(state, state.name), '=',
                  state.write0, dirmap[state.move0], renumber.get(state.next0, state.next0.name),
                  state.write1, dirmap[state.move1], renumber.get(state.next1, state.next1.name))

    def tm_print(self):
        """Prints the current state of the Turing machine execution."""
        tape = ''.join(' ' + x for x in self.left_tape) + \
            '[' + self.current_tape + ']' + ' '.join(reversed(self.right_tape))
        print('{state:{len}} {tape}'.format(len=self.longest_label, \
            state=self.state.name, tape=tape))

    def tm_step(self):
        """Executes the Turing machine for a single step."""
        self.tm_print()
        state = self.state

        if self.current_tape == '0':
            write, move, nextstate = state.write0, state.move0, state.next0
        else:
            write, move, nextstate = state.write1, state.move1, state.next1

        self.current_tape = write
        self.state = nextstate

        if move == 1:
            self.left_tape.append(self.current_tape)
            self.current_tape = self.right_tape.pop() if self.right_tape else '0'
        elif move == -1:
            self.right_tape.append(self.current_tape)
            self.current_tape = self.left_tape.pop() if self.left_tape else '0'
        else:
            assert False
