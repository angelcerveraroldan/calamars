//! Heap and garbage collector for Calamars

use std::ptr::NonNull;

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
    next_header: Option<NonNull<Header>>,
}

/// A wrapper around the memory that represented a heap object.
///
/// The memory is structured as: [ HEADER | PADDING? | PAYLOAD_MEMORY ]
#[repr(transparent)]
pub struct HeapObject(NonNull<Header>);

impl From<NonNull<Header>> for HeapObject {
    fn from(value: NonNull<Header>) -> Self {
        Self(value)
    }
}

impl HeapObject {
    fn header(&self) -> &Header {
        unsafe { self.0.as_ref() }
    }

    fn header_mut(&mut self) -> &mut Header {
        unsafe { self.0.as_mut() }
    }

    fn next_object(&self) -> Option<HeapObject> {
        let header = self.header();
        header.next_header.map(Into::into)
    }

    /// Get a raw pointer to the payload data
    fn _payload_ptr(&self, memlayout: &MemLayout) -> *mut u8 {
        let alignment = memlayout.alignment;
        let header_size = std::mem::size_of::<Header>();
        let header_size_with_padding = size_with_padding(header_size, alignment);
        let base = self.0.as_ptr() as *mut u8;
        unsafe { base.add(header_size_with_padding) }
    }

    /// Given a type to cast the payload into, do said casting and return an immutable reference to it
    fn _payload_ptr_as_type<T>(&self, memlayout: &MemLayout) -> *mut T {
        let ptr = self._payload_ptr(memlayout);
        ptr as *mut T
    }
}

pub struct Heap {
    head: Option<NonNull<Header>>,
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
