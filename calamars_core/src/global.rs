use crate::{
    StringArena,
    data_structs::{DStructArena, StructDefArena},
    ids,
    memory::MemoryLayoutArena,
    types::TypeArena,
};

pub struct FrontendCtx<'a> {
    pub types: &'a mut TypeArena,
    pub data_structs: &'a mut DStructArena,
    pub struct_defs: &'a mut StructDefArena,
    pub strings: &'a mut StringArena,
}

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
    pub fn frontend(&mut self) -> FrontendCtx<'_> {
        FrontendCtx {
            types: &mut self.types,
            data_structs: &mut self.data_structs,
            struct_defs: &mut self.struct_defs,
            strings: &mut self.strings,
        }
    }

    pub fn type_ctx(&mut self) -> TypeCtx<'_> {
        TypeCtx {
            types: &mut self.types,
            struct_defs: &self.struct_defs,
        }
    }
}
