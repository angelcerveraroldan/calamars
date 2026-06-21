//! Pre generated data for the MIR lowering
//!
//! The goal of this files is to generate file from the HIR that we
//! will later need to lower to MIR

use calamars_core::{
    data_structs::StructDef,
    global::GlobalContext,
    ids::{self, DStructId},
};
use front::sematic::hir;

pub type FieldIndex = usize;
pub type FieldName = String;

#[derive(Default)]
pub struct MirData {
    /// Given some data structure's field name, find its index
    struct_indices: hashbrown::HashMap<(DStructId, FieldName), FieldIndex>,
}

impl MirData {
    pub fn get_field_index_by_name(
        &self,
        dstructid: &DStructId,
        name: &str,
    ) -> Option<FieldIndex> {
        self.struct_indices
            .get(&(*dstructid, name.to_string()))
            .copied()
    }

    /// Given the HIR data of a struct, insert the data into the
    /// MirData table
    fn generate_struct_indices(
        &mut self,
        dstructid: ids::DStructId,
        ordered_fields: impl Iterator<Item = String>,
    ) -> crate::lower::MirRes<()> {
        for (index, field) in ordered_fields.enumerate() {
            let k =
                self.struct_indices.insert((dstructid, field.clone()), index);
            // This should never happen, if it does, then there is an
            // issue in the front end
            debug_assert!(
                k.is_none(),
                "Struct had the same key more than once"
            );
        }

        Ok(())
    }

    fn generate_struct_data(
        &mut self,
        dstructid: &DStructId,
        structure: &StructDef,
    ) -> crate::lower::MirRes<()> {
        let ordered_fields =
            structure.fields.iter().map(|field| field.name.clone());
        self.generate_struct_indices(*dstructid, ordered_fields)
    }

    pub fn generate_mirdata<'a>(hir_module: &'a hir::Module, global_ctx: &GlobalContext) -> Self {
        let mut s = Self::default();

        for dstructid in &hir_module.data_structs {
            let foo = global_ctx.struct_defs.get_unchecked(*dstructid);
            let _ = s.generate_struct_data(dstructid, foo);
        }

        s
    }
}
