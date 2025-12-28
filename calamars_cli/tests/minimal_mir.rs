use std::path::PathBuf;

use calamars_cli::source::SourceFile;
use calamars_core::ids;
use front::{
    sematic::{lower::HirBuilder, types::TypeHandler},
    syntax::parser::CalamarsParser,
};
use ir::lower::FunctionBuilder;

#[test]
/// A simple end to end test that checks that we can lower `minimal.cm` in the `docs/examples`
/// directory to MIR without any errors.
fn minimal_program_parses_types_and_lowers_to_mir() {
    // Locate the sample file
    let path = PathBuf::from("../docs/examples/minimal.cm");
    let sf = SourceFile::try_from((0, path)).expect("Failed to read minimal.cm");
    let tokens = sf.as_spanned_token_stream();

    let file_id = ids::FileId::from(0);
    let file_name = String::from("minimal.cm");

    // Parse
    let mut parser = CalamarsParser::new(file_id, tokens);
    let module = parser.parse_file();
    assert!(
        parser.diag().is_empty(),
        "Parser diagnostics found: {:?}",
        parser.diag()
    );

    // Lower AST -> HIR
    let (mut hir_module, hir_errors) =
        HirBuilder::lower_module(&module, file_id, file_name.clone());

    assert!(
        hir_errors.is_empty(),
        "HIR lowering produced errors: {:?}",
        hir_errors
    );

    // Type-check
    let mut type_handler = TypeHandler {
        module: &mut hir_module,
        errors: vec![],
    };
    type_handler.type_check_module();
    assert!(
        type_handler.errors.is_empty(),
        "Type-checking produced errors: {:?}",
        type_handler.errors
    );

    // Lower HIR -> MIR
    let mut mir_builder = FunctionBuilder::new(&hir_module);
    let mut functions = vec![];

    for symbolid in &hir_module.roots {
        let symbol = hir_module.symbols.get_unchecked(*symbolid);
        let name = symbol.ident_id();
        let return_ty = symbol.ty_id();

        let (params, body) =
            if let front::sematic::hir::SymbolKind::Function { params, body } = &symbol.kind {
                (params, body)
            } else {
                continue;
            };

        match mir_builder.lower(name, return_ty, params, *body) {
            Ok(fun) => functions.push(fun),
            Err(err) => panic!("Failed to lower a function: {:?}", err),
        }
    }

    assert!(
        !functions.is_empty(),
        "Expected at least one lowered function"
    );
}
