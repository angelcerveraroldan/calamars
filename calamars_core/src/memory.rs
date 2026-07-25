use crate::{UncheckedArena, ids};

pub type MemoryLayoutArena = UncheckedArena<MemLayout, ids::MemLayoutId>;

/// A types layout informaton, all sizes in bytes
pub struct FieldMemInfo {
    pub size: usize,
    pub align: usize,
    pub is_pointer: bool,
}

/// A descriptor table for memory layouts
pub struct MemLayout {
    /// Total size
    ///
    /// If this memlayout contains sub elements (i.e. a struct) and there is padding
    /// between them, then this padding will be part of this size, padding between
    /// the header and the object will not be part of this size.
    ///
    /// If the size is not know at compile time, then we will set this to None.
    pub comp_size: Option<usize>,
    pub alignment: usize,
    /// Memory information about each of the fields
    pub field_info: Box<[FieldMemInfo]>,
    //// The offset of each of the fields
    pub offsets: Box<[usize]>,
}

impl MemLayout {
    fn string() -> Self {
        MemLayout {
            comp_size: None,
            alignment: std::mem::align_of::<usize>(),
            field_info: [].into(),
            offsets: [].into(),
        }
    }

    pub fn const_id_string() -> ids::MemLayoutId {
        ids::MemLayoutId::from(0)
    }
}

impl Default for MemoryLayoutArena {
    fn default() -> Self {
        let mut data: Vec<MemLayout> = Default::default();
        data.push(MemLayout::string()); // String -> id of 0
        Self {
            data,
            _pd: Default::default(),
        }
    }
}
