//! Heap and garbage collector for Calamars

/// State used to keep track of which pointers are reachable from roots
///
/// The default value *has to be* 0, so that we can delete pointers from the heap while
/// the programme is running concurrently (while keeping memory safety). If a new pointer
/// is created, we will not delete it, we assume that it is reachable.
#[derive(Default)]
pub enum SweepState {
    /// Pointer has been reached, and deletion would lead to memory safety issues
    #[default]
    Marked = 0,
    /// Pointer has not been reached, it is safe to delete
    Unmarked = 1,
}

/// An identifier to the MemoryLayoutId, which will be re-used by many pointers
#[derive(Debug, Copy, Clone)]
pub struct MemLayoutId(usize);

/// A descriptor table for memory layouts
pub struct MemLayout {
    /// Total size (including padding)
    ///
    /// This includes the padding, but does not need to be a power of two, that will be
    /// handled when allocating.
    size: usize,
    alignment: usize,
    /// Offsets to each of the pointers.
    ///
    /// This will help traverse pointers when doing sweep.
    pub pointer_offsets: Box<[usize]>,
}

pub struct Header {
    /// What sort of pointer is this
    memtag: MemoryTag,
    /// Information about the memory layout
    memlayout_id: MemLayoutId,
    marked: SweepState,
    /// A pointer to the next header
    ///
    /// Will be None if this is the last header.
    next_header: Option<std::ptr::NonNull<Header>>,
}

/// A wrapper around the memory that represented a heap object.
///
/// The memory is structured as: [ HEADER | PADDING? | PAYLOAD_MEMORY ]
pub struct HeapObject {
    memory: *mut u8,
}

impl HeapObject {
    fn read_header_as_mut_ptr(&self) -> *mut Header {
        self.memory as *mut Header
    }

    /// Read the header that is saved in the raw memory
    pub fn read_header(&self) -> &Header {
        unsafe {
            let mheader = self.read_header_as_mut_ptr();
            &*mheader
        }
    }

    /// Return a pointer to the payload of this heap object
    pub fn payload_prt(&self) -> *mut u8 {
        todo!()
    }
}

pub struct Heap {
    object_list: HeapObject,
}

pub enum MemoryTag {
    Liteal,  // nothing else to look at (basic types)
    String,  // nothing else to look at
    Struct,  // look at children
    Enum,    // look at children
    Closure, // for closures we need to look at what has been captured
}

#[rustfmt::skip]
fn size_with_padding(size: usize, alignment: usize) -> usize {
    let diff = size % alignment;
	if diff == 0 { size } else { size + alignment - diff }
}

fn allocate_memory(memlayout: &MemLayout) -> *mut u8 {
    let header_size = std::mem::size_of::<Header>();
    let alignment = std::mem::align_of::<Header>().max(memlayout.alignment);
    // We need to make sure that the memory for the payload is aligned correctly, thus we will
    // align to its desired alignment
    let total_size = size_with_padding(header_size, memlayout.alignment) + memlayout.size;
    unsafe {
        let l = std::alloc::Layout::from_size_align_unchecked(total_size, alignment);
        std::alloc::alloc(l)
    }
}
