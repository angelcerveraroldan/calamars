//! Heap and garbage collector for Calamars

use std::ptr::NonNull;

use calamars_core::{global::GlobalContext, ids, memory::MemLayout};

use crate::errors::{VError, VResult};

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

pub struct Header {
    /// What sort of pointer is this
    memtag: MemoryTag,
    /// Information about the memory layout
    memlayout_id: ids::MemLayoutId,
    marked: SweepState,
    /// A pointer to the next header
    ///
    /// Will be None if this is the last header.
    next_header: Option<HeapObject>,
}

impl Header {
    pub fn new(
        memtag: MemoryTag,
        memlayout_id: ids::MemLayoutId,
        marked: SweepState,
        next_header: Option<HeapObject>,
    ) -> Self {
        Self {
            memtag,
            memlayout_id,
            marked,
            next_header,
        }
    }
}

/// A wrapper around the memory that represented a heap object.
///
/// The memory is structured as: [ HEADER | PADDING? | PAYLOAD_MEMORY ]
pub type HeapObject = NonNull<Header>;

fn new_detached_object_struct(
    ctx: &GlobalContext,
    memtag: MemoryTag,
    memlayout_id: ids::MemLayoutId,
) -> VResult<HeapObject> {
    debug_assert!({
        let memlayout = ctx.memlay.get_unchecked(memlayout_id);
        memlayout.comp_size.is_some()
    });
    _new_detached_object(ctx, memtag, None, memlayout_id)
}

fn new_detached_object_string(
    ctx: &GlobalContext,
    memtag: MemoryTag,
    len: usize,
    memlayout_id: ids::MemLayoutId,
) -> VResult<HeapObject> {
    debug_assert!({
        let memlayout = ctx.memlay.get_unchecked(memlayout_id);
        memlayout.comp_size.is_none()
    });
    let size = std::mem::size_of::<usize>() + len;
    _new_detached_object(ctx, memtag, Some(size), memlayout_id)
}

fn _new_detached_object(
    ctx: &GlobalContext,
    memtag: MemoryTag,
    // for instances where the size is run-time
    runtime_size: Option<usize>,
    memlayout_id: ids::MemLayoutId,
) -> VResult<HeapObject> {
    let memlayout = ctx.memlay.get_unchecked(memlayout_id);

    if runtime_size.is_some() && memlayout.comp_size.is_some() {
        return Err(VError::HeapError(
            "Cannot pass both a runtime and a compile time size",
        ));
    }
    let Some(size) = runtime_size.or(memlayout.comp_size) else {
        return Err(VError::HeapError(
            "When compile time size is not know, a runtime size must be passed to the allocator",
        ));
    };
    let memory = allocate_memory(memlayout.alignment, size);
    if memory.is_null() {
        return Err(VError::HeapError("Null memory was returned"));
    }
    let header = Header::new(memtag, memlayout_id, SweepState::Unmarked, None);
    // Now we can save the raw data to the pointer
    let nnheader = unsafe {
        let header_ptr = memory as *mut Header;
        std::ptr::write(header_ptr, header);
        std::ptr::NonNull::new_unchecked(header_ptr)
    };
    Ok(HeapObject::from(nnheader))
}

#[inline(always)]
fn header(heap_obj: &HeapObject) -> &Header {
    unsafe { heap_obj.as_ref() }
}

#[inline(always)]
fn header_mut(heap_obj: &mut HeapObject) -> &mut Header {
    unsafe { heap_obj.as_mut() }
}

#[inline(always)]
fn set_next_header(heap_obj: &mut HeapObject, next: Option<HeapObject>) {
    header_mut(heap_obj).next_header = next;
}

#[inline(always)]
fn next_object(heap_obj: &HeapObject) -> Option<HeapObject> {
    let header = header(heap_obj);
    header.next_header.map(Into::into)
}

/// Get a raw pointer to the payload data
fn _payload_ptr(heap_obj: &HeapObject, alignment: usize) -> *mut u8 {
    let header_size = std::mem::size_of::<Header>();
    let header_size_with_padding = size_with_padding(header_size, alignment);
    let base = heap_obj.as_ptr() as *mut u8;
    unsafe { base.add(header_size_with_padding) }
}

/// Given a type to cast the payload into, do said casting and return an immutable reference to it
fn _payload_ptr_as_type<T>(heap_obj: &HeapObject, alignment: usize) -> *mut T {
    let ptr = _payload_ptr(heap_obj, alignment);
    ptr as *mut T
}

fn save_data_to_payload<T>(heap_obj: &mut HeapObject, src: T, alignment: usize) {
    let ptr = _payload_ptr_as_type::<T>(heap_obj, alignment);
    unsafe { std::ptr::write(ptr, src) }
}

pub struct Heap {
    head: Option<HeapObject>,
}

impl Heap {
    pub fn alloca_struct() -> VResult<()> {
        todo!("Leaving this here, but it will be implemented after strings")
    }

    pub fn alloca_string(&mut self, ctx: &GlobalContext, string_id: ids::StringId) -> VResult<()> {
        let string = ctx.strings.get_unchecked(string_id);
		let slen = string.len();
        let memid = MemLayout::const_id_string();
        let mut str_obj = new_detached_object_string(ctx, MemoryTag::String, slen, memid)?;
        header_mut(&mut str_obj).next_header = self.head;
        unsafe {
            // Save the length of the string
            let ptr = _payload_ptr_as_type::<usize>(&mut str_obj, std::mem::align_of::<usize>());
            *ptr = string.len();
            // Save the actual string
            let ptr = ptr.add(1) as *mut u8;
            let str_ptr = string.as_ptr();
            std::ptr::copy_nonoverlapping(str_ptr, ptr, slen);
        }
        self.head = Some(str_obj);
        Ok(())
    }
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

/// Given some needed size, and some desired alignment, allocate memory
fn allocate_memory(alignment: usize, size: usize) -> *mut u8 {
    let header_size = std::mem::size_of::<Header>();
    let alignment = std::mem::align_of::<Header>().max(alignment);
    // We need to make sure that the memory for the payload is aligned correctly, thus we will
    // align to its desired alignment
    let total_size = size_with_padding(header_size, alignment) + size;
    unsafe {
        let l = std::alloc::Layout::from_size_align_unchecked(total_size, alignment);
        std::alloc::alloc(l)
    }
}
