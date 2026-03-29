use crate::ids;

pub type DStructArena = crate::InternArena<DataStructure, ids::DStructId>;

/// A key for hasing new data structures defined in calamars
/// This key can be used to lookup and generates ids, which can then be used to
/// retrieve data about the structure.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct DataStructure {
    // The name where the structure is declared
    pub name: String,
    pub module: ids::FileId,
}
