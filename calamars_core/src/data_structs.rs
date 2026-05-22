use crate::ids;

pub type DStructArena = crate::InternArena<DataStructureKey, ids::DStructId>;
pub type StructDefArena = crate::UncheckedArena<StructDef, ids::DStructId>;

/// A key for hasing new data structures defined in calamars
/// This key can be used to lookup and generates ids, which can then be used to
/// retrieve data about the structure.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct DataStructureKey {
    // The name where the structure is declared
    pub name: String,
    pub module: ids::FileId,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct StructFieldDef {
    pub name: String,
    pub ty: ids::TypeId,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct StructDef {
    pub key: DataStructureKey,
    pub fields: Box<[StructFieldDef]>,
}

impl StructDef {
    pub fn field(&self, field_name: &str) -> Option<&StructFieldDef> {
        self.fields.iter().find(|field| field.name == field_name)
    }
}
