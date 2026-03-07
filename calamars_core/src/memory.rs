use crate::{UncheckedArena, ids};

pub type MemoryLayoutArena = UncheckedArena<MemLayout, ids::MemLayoutId>;

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
    /// Offsets to each of the pointers.
    ///
    /// This will help traverse pointers when doing sweep.
    pub pointer_offsets: Box<[usize]>,
}

impl MemLayout {
    fn string() -> Self {
        MemLayout {
            comp_size: None,
            alignment: std::mem::align_of::<usize>(),
            pointer_offsets: [].into(),
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
