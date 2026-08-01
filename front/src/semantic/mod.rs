use calamars_core::{
    StringArena,
    data_structs::{DStructArena, StructDefArena},
    global::GlobalContext,
    types::TypeArena,
};

pub mod error;
pub mod hir;
pub mod lower;
pub mod types;

pub struct FrontendCtx<'a> {
    pub types: &'a mut TypeArena,
    pub data_structs: &'a mut DStructArena,
    pub struct_defs: &'a mut StructDefArena,
    pub strings: &'a mut StringArena,
}

impl<'a> From<&'a mut GlobalContext> for FrontendCtx<'a> {
    fn from(value: &'a mut GlobalContext) -> Self {
        FrontendCtx {
            types: &mut value.types,
            data_structs: &mut value.data_structs,
            struct_defs: &mut value.struct_defs,
            strings: &mut value.strings,
        }
    }
}
