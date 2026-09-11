//! Pre generated data for the MIR lowering
//!
//! The goal of this files is to generate file from the HIR that we
//! will later need to lower to MIR

use calamars_core::{
    data_structs::StructDef,
    global::GlobalContext,
    ids::{self, DStructId},
};
use front::semantic::hir;

pub type FieldIndex = usize;
pub type FieldName = String;

pub struct FieldInfo {
    pub name: String,
    pub ftype: ids::TypeId,
    pub index: FieldIndex,
}

#[derive(Default)]
pub struct MirData {
    /// Given some data structure's field name, find its index
    struct_indices: hashbrown::HashMap<(DStructId, FieldName), FieldInfo>,
}

impl MirData {
    pub fn get_field_index_by_name(&self, dstructid: &DStructId, name: &str) -> Option<FieldIndex> {
        self.struct_indices
            .get(&(*dstructid, name.to_string()))
            .map(|field| field.index)
    }

    pub fn get_field_info_by_name(&self, dstructid: &DStructId, name: &str) -> Option<&FieldInfo> {
        self.struct_indices.get(&(*dstructid, name.to_string()))
    }

    fn generate_struct_data(
        &mut self,
        dstructid: &DStructId,
        structure: &StructDef,
    ) -> crate::lower::MirRes<()> {
        for (index, field) in structure.fields.iter().enumerate() {
            let k = self.struct_indices.insert(
                (*dstructid, field.name.clone()),
                FieldInfo {
                    name: field.name.clone(),
                    ftype: field.ty,
                    index,
                },
            );
            // This should never happen, if it does, then there is an
            // issue in the front end
            debug_assert!(k.is_none(), "Struct had the same key more than once");
        }
        Ok(())
    }

    pub fn generate_mirdata<'a>(
        hir_tmodule: &'a hir::TypedModule,
        global_ctx: &GlobalContext,
    ) -> Self {
        let mut s = Self::default();

        for dstructid in &hir_tmodule.hir.data_structs {
            let foo = global_ctx.struct_defs.get_unchecked(*dstructid);
            let _ = s.generate_struct_data(dstructid, foo);
        }

        s
    }
}
