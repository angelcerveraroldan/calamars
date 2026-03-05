use crate::{StringArena, types::TypeArena};

/// Context and information that needs to be filled and
/// shared between many stages of the compiler
pub struct GlobalContext {
    pub types: TypeArena,
    pub strings: StringArena,
}
