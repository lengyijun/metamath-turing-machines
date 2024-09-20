from collections import namedtuple

class Node:
    """Base class for all Not Quite Laconic syntax nodes."""
    def __init__(self, **kwargs):
        self.lineno = kwargs.pop('lineno', 0)
        self.children = kwargs.pop('children', [])
        assert not kwargs
        self.check_children()

    child_types = ()

    def check_children(self):
        """Verifies that the node has the correct number and types of child
        nodes."""
        if isinstance(self.child_types, tuple):
            assert len(self.children) == len(self.child_types)
            for child, ctype in zip(self.children, self.child_types):
                assert isinstance(child, ctype)
        else:
            for child in self.children:
                assert isinstance(child, self.child_types)

    def error(self, message):
        """Print an error using the line number of this node."""
        raise str(self.lineno) + ": " + message

    repr_suppress = ('lineno','children')

    def __repr__(self):
        result = []
        result.append(self.__class__.__name__ + '(')
        result.append(('\n  ',''))

        has_items = False
        for k, v in vars(self).items():
            if k in self.repr_suppress:
                continue
            result.append(k + '=' + repr(v).replace('\n', '\n  '))
            result.append((',\n  ', ', '))
            has_items = True

        if self.children:
            result.append('children=[')
            result.append(('\n    ', ''))
            for child in self.children:
                result.append(repr(child).replace('\n', '\n    '))
                result.append((',\n    ', ', '))
            result.pop()
            result.append(']')
        elif has_items:
            result.pop()
        result.append(')')

        result = [(tup if isinstance(tup, tuple) else (tup, tup)) for tup in result]
        broken = ''.join(a for a, b in result)
        unbroken = ''.join(b for a, b in result)
        if len(unbroken) < 80 and '\n' not in unbroken:
            return unbroken
        else:
            return broken

class NatExpr(Node):
    """Base class for expressions which result in a natural number.

    Sub classes should define an emit method which generates code to put the
    evaluation result in a caller-allocated temporary register.

    TODO: context-sensitive code generation and peephole optimization will
    reduce the state count here quite a bit."""

    def emit_nat(self, state, target):
        """Calculate the value of this expression into the target register,
        which is guaranteed to be zero by the caller unless is_additive
        returns True."""
        temps = []
        for child in self.children:
            temp = state.get_temp()
            temps.append(temp)
            child.emit_nat(state, temp)
        self.emit_nat_op(state, target, temps)
        for temp in temps:
            state.put_temp(temp)

    def emit_nat_add(self, state, out):
        if self.is_additive():
            self.emit_nat(state, out)
        else:
            temp = state.get_temp()
            self.emit_nat(state, temp)
            state.emit_transfer(temp, out)
            state.put_temp(temp)

    def emit_nat_op(self, state, target, temps):
        """Calculate the value of this expression with the arguments already
        evaluated.

        To customize argument evaluation, override emit_nat instead."""
        raise NotImplementedError()

    def is_additive(self):
        """Returns True if emit_nat actually just adds and is safe for non-zero targets."""
        return False

class Reg(NatExpr):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        super().__init__(**kwargs)

    def is_additive(self):
        return True

    def emit_nat_op(self, state, target, _args):
        save = state.get_temp()
        reg = state.resolve(self.name)
        state.emit_transfer(reg, target, save)
        state.emit_transfer(save, reg)
        state.put_temp(save)

class Mul(NatExpr):
    child_types = (NatExpr, NatExpr)
    def is_additive(self):
        return True

    def emit_nat(self, state, out):
        lhs_ex, rhs_ex = self.children
        if lhs_ex.is_additive() and not rhs_ex.is_additive():
            lhs_ex, rhs_ex = rhs_ex, lhs_ex
        lhs = state.get_temp()
        lhs_ex.emit_nat(state, lhs)
        again = state.gensym()
        done = state.gensym()
        state.emit_label(again)
        state.emit_dec(lhs)
        state.emit_goto(done)
        rhs_ex.emit_nat_add(state, out)
        state.emit_goto(again)
        state.emit_label(done)
        state.put_temp(lhs)

class Div(NatExpr):
    child_types = (NatExpr, NatExpr)
    def is_additive(self):
        return True

    def emit_nat(self, state, out):
        dividend_ex, divisor_ex = self.children

        dividend = state.get_temp()
        divisor = state.get_temp()

        loop_quotient = state.gensym()
        loop_divisor = state.gensym()
        exhausted = state.gensym()
        full_divisor = state.gensym()

        dividend_ex.emit_nat(state, dividend)

        state.emit_label(loop_quotient)
        divisor_ex.emit_nat(state, divisor)
        state.emit_label(loop_divisor)
        state.emit_dec(divisor)
        state.emit_goto(full_divisor)
        state.emit_dec(dividend)
        state.emit_goto(exhausted)
        state.emit_goto(loop_divisor)
        state.emit_label(full_divisor)

        state.emit_inc(out)
        state.emit_goto(loop_quotient)
        state.emit_label(exhausted)
        state.emit_transfer(divisor)

        state.put_temp(dividend)
        state.put_temp(divisor)

class Add(NatExpr):
    child_types = NatExpr
    def is_additive(self):
        return True

    def emit_nat(self, state, out):
        for child in self.children:
            child.emit_nat_add(state, out)

class Lit(NatExpr):
    def __init__(self, **kwargs):
        self.value = kwargs.pop('value')
        super().__init__(**kwargs)

    def is_additive(self):
        return True

    def emit_nat_op(self, state, out, _args):
        for _ in range(self.value):
            state.emit_inc(out)

class Monus(NatExpr):
    """Subtracts the right argument from the left argument, clamping to zero
    (also known as the "monus" operator)."""
    child_types = (NatExpr, NatExpr)

    def emit_nat_op(self, state, out, args):
        lhs, rhs = args
        # TODO: forward directly out to lhs
        state.emit_transfer(lhs, out)
        loop = state.gensym()
        done = state.gensym()
        state.emit_label(loop)
        state.emit_dec(rhs)
        state.emit_goto(done)
        state.emit_dec(out)
        state.emit_noop()
        state.emit_goto(loop)
        state.emit_label(done)

class BoolExpr(Node):
    """Base class for expressions which result in a boolean test."""

    def emit_test(self, state, target, invert):
        """Evaluate the test and jump to label if the test is true, subject to
        the inversion flag."""
        temps = []
        for child in self.children:
            temp = state.get_temp()
            temps.append(temp)
            child.emit_nat(state, temp)
        self.emit_test_op(state, target, invert, temps)
        for temp in temps:
            state.put_temp(temp)

    def emit_test_op(self, state, target, invert, temps):
        """Calculate the value of this test with the arguments already
        evaluated.

        To customize argument evaluation, override emit_test instead."""
        raise NotImplementedError()

class CompareBase(BoolExpr):
    child_types = (NatExpr, NatExpr)
    jump_lt = False
    jump_eq = False
    jump_gt = False

    def emit_compare_reg_0(self, state, label, j_eq, j_gt, name):
        # LT is not possible here

        no_jump = state.gensym()
        state.emit_dec(state.resolve(name))
        state.emit_goto(label if j_eq else no_jump)
        state.emit_inc(state.resolve(name))
        state.emit_goto(label if j_gt else no_jump)
        state.emit_label(no_jump)

    def emit_compare_lit(self, state, label, j_lt, j_eq, j_gt, lhs_ex, rhs_val):
        if isinstance(lhs_ex, Reg) and rhs_val == 0:
            return self.emit_compare_reg_0(state, label, j_eq, j_gt, lhs_ex.name)

        lhs = state.get_temp()
        lhs_ex.emit_nat(state, lhs)

        no_jump = state.gensym()
        for _ in range(rhs_val):
            state.emit_dec(lhs)
            state.emit_goto(label if j_lt else no_jump)

        if j_eq != j_gt:
            state.emit_dec(lhs)
            state.emit_goto(label if j_eq else no_jump)

        state.emit_transfer(lhs)
        if j_gt:
            state.emit_goto(label)
        state.emit_label(no_jump)
        state.put_temp(lhs)

    def emit_test(self, state, label, invert):
        lhs_ex, rhs_ex = self.children

        jump_lt, jump_eq, jump_gt = self.jump_lt ^ invert, self.jump_eq ^ invert, \
            self.jump_gt ^ invert

        if isinstance(rhs_ex, Lit):
            return self.emit_compare_lit(state, label, jump_lt, jump_eq, jump_gt, lhs_ex, rhs_ex.value)
        if isinstance(lhs_ex, Lit):
            return self.emit_compare_lit(state, label, jump_gt, jump_eq, jump_lt, rhs_ex, lhs_ex.value)

        lhs = state.get_temp()
        lhs_ex.emit_nat(state, lhs)
        rhs = state.get_temp()
        rhs_ex.emit_nat(state, rhs)

        monus = state.gensym()
        not_less = state.gensym()
        is_less = state.gensym()
        no_jump = state.gensym()

        state.emit_label(monus)
        state.emit_dec(rhs)
        state.emit_goto(not_less)
        state.emit_dec(lhs)
        state.emit_goto(is_less)
        state.emit_goto(monus)

        state.emit_label(not_less)
        if jump_eq != jump_gt:
            state.emit_dec(lhs)
            state.emit_goto(label if jump_eq else no_jump)
        state.emit_transfer(lhs)
        state.emit_goto(label if jump_gt else no_jump)

        state.emit_label(is_less)
        state.emit_transfer(rhs)
        state.emit_goto(label if jump_lt else no_jump)

        state.emit_label(no_jump)

        state.put_temp(lhs)
        state.put_temp(rhs)

class Less(CompareBase):
    jump_lt = True

class LessEqual(CompareBase):
    jump_lt = True
    jump_eq = True

class Greater(CompareBase):
    jump_gt = True

class GreaterEqual(CompareBase):
    jump_eq = True
    jump_gt = True

class Equal(CompareBase):
    jump_eq = True

class NotEqual(CompareBase):
    jump_lt = True
    jump_gt = True

class Not(BoolExpr):
    child_types = (BoolExpr,)

    def emit_test(self, state, label, invert):
        self.children[0].emit_test(state, label, not invert)

class And(BoolExpr):
    child_types = (BoolExpr,BoolExpr)
    is_or = False

    def emit_test(self, state, label, invert):
        left, right = self.children
        if invert ^ self.is_or:
            left.emit_test(state, label, True ^ self.is_or)
            right.emit_test(state, label, True ^ self.is_or)
        else:
            dont_jump = state.gensym()
            left.emit_test(state, dont_jump, True ^ self.is_or)
            right.emit_test(state, label, False ^ self.is_or)
            state.emit_label(dont_jump)

class Or(And):
    is_or = True

class BoolConst(BoolExpr):
    def emit_test(self, state, label, invert):
        if self.value ^ invert:
            state.emit_goto(label)

class TrueConst(BoolConst):
    value = True

class FalseConst(BoolConst):
    value = False

class VoidExpr(Node):
    """Base class for expressions which return no value."""

    def emit_stmt(self, state):
        raise NotImplementedError()

class Assign(VoidExpr):
    child_types = (Reg, NatExpr)
    # TODO: augmented additions and subtractions can be peepholed to remove the temporary
    # TODO: when assigning something that doesn't use the old value, it can be constructed in place

    def emit_aug_op(self, state, lhs, rhs):
        if not (isinstance(rhs, Add) or isinstance(rhs, Monus)):
            return
        if len(rhs.children) != 2:
            return
        rhs_l, rhs_r = rhs.children
        if not (isinstance(rhs_l, Reg) and rhs_l.name == lhs.name):
            return
        if not isinstance(rhs_r, Lit):
            return
        for _ in range(rhs_r.value):
            if isinstance(rhs, Monus):
                state.emit_dec(state.resolve(lhs.name))
                state.emit_noop()
            else:
                state.emit_inc(state.resolve(lhs.name))
        return True

    def emit_stmt(self, state):
        lhs, rhs = self.children
        if isinstance(rhs, Lit):
            state.emit_transfer(state.resolve(lhs.name))
            rhs.emit_nat(state, state.resolve(lhs.name))
        elif self.emit_aug_op(state, lhs, rhs):
            pass
        else:
            temp = state.get_temp()
            rhs.emit_nat(state, temp)
            state.emit_transfer(state.resolve(lhs.name))
            state.emit_transfer(temp, state.resolve(lhs.name))
            state.put_temp(temp)

class Block(VoidExpr):
    child_types = VoidExpr
    def emit_stmt(self, state):
        for st in self.children:
            st.emit_stmt(state)

class WhileLoop(VoidExpr):
    child_types = (BoolExpr, VoidExpr)
    def emit_stmt(self, state):
        test, block = self.children
        exit = state.gensym()
        again = state.gensym()
        state.emit_label(again)
        test.emit_test(state, exit, True)
        block.emit_stmt(state)
        state.emit_goto(again)
        state.emit_label(exit)

class IfThen(VoidExpr):
    child_types = (BoolExpr, VoidExpr, VoidExpr)
    def emit_stmt(self, state):
        test, then_, else_ = self.children
        l_else = state.gensym()
        l_then = state.gensym()
        test.emit_test(state, l_else, True)
        then_.emit_stmt(state)
        state.emit_goto(l_then)
        state.emit_label(l_else)
        else_.emit_stmt(state)
        state.emit_label(l_then)

class SwitchArm(Block):
    def __init__(self, **kwargs):
        self.case = kwargs.pop('case')
        assert self.case is None or isinstance(self.case, int) and self.case >= 0
        super().__init__(**kwargs)

class Break(VoidExpr):
    def emit_stmt(self, state):
        assert state.break_label
        state.emit_goto(state.break_label)

class Switch(VoidExpr):
    def check_children(self):
        head, *arms = self.children
        assert isinstance(head, NatExpr)
        for arm in arms:
            assert isinstance(arm, SwitchArm)

    def emit_stmt(self, state):
        head_ex, *arms_ex = self.children

        head = state.get_temp()
        head_ex.emit_nat(state, head)

        arm_labels = {}
        for arm in arms_ex:
            if arm.case is None or arm.case in arm_labels:
                continue
            arm_labels[arm.case] = state.gensym()

        default_label = state.gensym()

        for count in range(max(arm_labels)):
            state.emit_dec(head)
            state.emit_goto(arm_labels.get(count, default_label))

        state.emit_transfer(head)
        state.emit_goto(default_label)
        state.put_temp(head)

        save_break_label, state.break_label = state.break_label, state.gensym()

        for arm in arms_ex:
            if arm.case is None:
                assert default_label
                state.emit_label(default_label)
                arm.emit_stmt(state)
                default_label = None
            else:
                assert arm.case in arm_labels
                state.emit_label(arm_labels.pop(arm.case))
                arm.emit_stmt(state)

        if default_label:
            state.emit_label(default_label)
        state.emit_label(state.break_label)
        state.break_label = save_break_label

class Call(VoidExpr):
    child_types = Reg
    def __init__(self, **kwargs):
        self.func = kwargs.pop('func')
        super().__init__(**kwargs)

    def emit_stmt(self, state):
        state.emit_call(self.func, [state.resolve(arg.name) for arg in self.children])

class Return(VoidExpr):
    def emit_stmt(self, state):
        state.emit_return()

class GlobalNode(Node):
    pass

class ProcDef(GlobalNode):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.parameters = kwargs.pop('parameters')
        super().__init__(**kwargs)

    child_types = (VoidExpr,)

class GlobalReg(GlobalNode):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        super().__init__(**kwargs)

class Program(Node):
    child_types = GlobalNode
    repr_suppress = Node.repr_suppress + ('by_name',)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.by_name = {node.name: node for node in self.children}
