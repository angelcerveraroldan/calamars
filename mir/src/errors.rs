#[derive(Debug)]
pub enum MirErrors {
    LoweringErr { msg: String },
    NoWorkingBlock,
    IdentNotFound,
    ExpressionNotFound,
    CouldNotGetExpressionType,
    ParamNotFound,
}
