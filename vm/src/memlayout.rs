use calamars_core::{
    data_structs::StructDef,
    global::GlobalContext,
    ids::{DStructId, TypeId},
    memory::{FieldKindTy, FieldMemInfo, MemLayout},
    types::TypeArena,
};

use crate::heap::HeapObject;

fn type_layout(ty: TypeId, types: &TypeArena) -> FieldMemInfo {
    let ty = types.get_unchecked(ty);
    match ty {
        calamars_core::types::Type::Error => {
            unreachable!("We should not have made it past the HIR with error types")
        }
        calamars_core::types::Type::Char => FieldMemInfo {
            size: 4,
            align: 4,
            is_pointer: false,
            kind: FieldKindTy::Char,
        },
        calamars_core::types::Type::Boolean => FieldMemInfo {
            size: 1,
            align: 1,
            is_pointer: false,
            kind: FieldKindTy::Boolean,
        },
        calamars_core::types::Type::Unit => FieldMemInfo {
            size: 1,
            align: 1,
            is_pointer: false,
            kind: FieldKindTy::Unit,
        },
        calamars_core::types::Type::Integer => FieldMemInfo {
            size: 8,
            align: 8,
            is_pointer: false,
            kind: FieldKindTy::Integer,
        },
        calamars_core::types::Type::Float => FieldMemInfo {
            size: 8,
            align: 8,
            is_pointer: false,
            kind: FieldKindTy::Float,
        },
        calamars_core::types::Type::String => FieldMemInfo {
            size: size_of::<HeapObject>(),
            align: align_of::<HeapObject>(),
            is_pointer: true,
            kind: FieldKindTy::StringPtr,
        },
        calamars_core::types::Type::Structure(_) => FieldMemInfo {
            size: size_of::<HeapObject>(),
            align: align_of::<HeapObject>(),
            is_pointer: true,
            kind: FieldKindTy::StructPtr,
        },
        _ => todo!("memory info for this is not supported"),
    }
}

/// Fill the global context with memory layout information
pub fn generate_structs_mem_layout(gctx: &mut GlobalContext) {
    // TODO: We shuold implement an iterator here ...
    for (raw_id, struct_def) in gctx.struct_defs.inner().iter().enumerate() {
        let mem_layout = struct_layout(struct_def, &gctx.types);
        let mem_layout_id = gctx.memlay.push(mem_layout);
        gctx.struct_mem
            .insert(DStructId::from(raw_id), mem_layout_id);
    }
}

/// Given a structs' information, generate its memeory layout
pub fn struct_layout(struct_def: &StructDef, types: &TypeArena) -> MemLayout {
    let mut largest_alignment = 0;
    let mut comp_size = 0; // total size needed
    let mut field_info = vec![];
    let mut field_offset = vec![];
    for field in &struct_def.fields {
        let layout: FieldMemInfo = type_layout(field.ty, types);
        largest_alignment = largest_alignment.max(layout.align);

        // add padding if needed
        let f = comp_size % layout.align;
        if f != 0 {
            comp_size += layout.align - f;
        }
        field_offset.push(comp_size);
        comp_size += layout.size;
        field_info.push(layout);
    }
    // Add tail padding if needed
    let f = comp_size % largest_alignment;
    if f != 0 {
        comp_size += largest_alignment - f;
    }

    // Empty structs should be removed after type check
    debug_assert_ne!(largest_alignment, 0);

    MemLayout {
        comp_size: Some(comp_size),
        alignment: largest_alignment,
        field_info: field_info.into(),
        offsets: field_offset.into(),
    }
}
