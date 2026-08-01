use crate::{
    StringArena,
    data_structs::{DStructArena, StructDefArena},
    ids,
    memory::MemoryLayoutArena,
    types::TypeArena,
};

pub struct TypeCtx<'a> {
    pub types: &'a mut TypeArena,
    pub struct_defs: &'a StructDefArena,
}

/// Context and information that needs to be filled and
/// shared between many stages of the compiler
pub struct GlobalContext {
    pub types: TypeArena,
    pub data_structs: DStructArena,
    pub struct_defs: StructDefArena,
    pub strings: StringArena,
    pub memlay: MemoryLayoutArena,
    pub struct_mem: hashbrown::HashMap<ids::DStructId, ids::MemLayoutId>,
}

impl GlobalContext {
    pub fn type_ctx(&mut self) -> TypeCtx<'_> {
        TypeCtx {
            types: &mut self.types,
            struct_defs: &self.struct_defs,
        }
    }
}
