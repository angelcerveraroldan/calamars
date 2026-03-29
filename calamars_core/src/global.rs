use crate::{StringArena, data_structs::DStructArena, memory::MemoryLayoutArena, types::TypeArena};

/// Context and information that needs to be filled and
/// shared between many stages of the compiler
pub struct GlobalContext {
    pub types: TypeArena,
    pub data_structs: DStructArena,
    pub strings: StringArena,
    pub memlay: MemoryLayoutArena,
}
