#[derive(Debug)]
pub enum MirErrors {
    LoweringErr { msg: String },
    NoWorkingBlock,
    IdentNotFound,
    InstNotFound,
    ExpressionNotFound,
    CouldNotGetExpressionType,
    ParamNotFound,
    InstructionListWasEmpty,
}
