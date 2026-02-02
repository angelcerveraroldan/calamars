pub type VResult<A> = Result<A, VError>;

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum VError {
    RegisterOutOfBounds {
        index: u32,
        len: u32,
    },
    BytecodeOutOfBounds {
        index: u32,
        len: u32,
    },
    BlockNotFound {
        block: u32,
    },
    PhiMissingIncoming {
        block: u32,
    },
    TypeMismatchBinary {
        lhs: &'static str,
        rhs: &'static str,
    },
    TypeMismatchUnary {
        op: &'static str,
        found: &'static str,
    },
    InvalidConditionType {
        found: &'static str,
    },
    InvalidReturnValue,
    UnsupportedConstant,
    UnsupportedExtern,
    UnsupportedInstruction,
    InternalInstructionNotFound,
    UninitializedRegister {
        index: u32,
    },
    FunctionNotFound {
        id: u32,
    },
    EmptyStack,
}
