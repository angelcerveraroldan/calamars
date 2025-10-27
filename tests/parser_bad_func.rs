use calamars::{
    parser::parse_module,
    syntax::{
        ast::{Declaration, Type},
        token::Token,
    },
};
use chumsky::Parser;
use std::fs;

#[test]
fn parser_bad_fn() {
    let source = fs::read_to_string("tests/test_files/bad_function.cm").unwrap();
    let stream = Token::tokens_spanned_stream(&source);
    let (out, errs) = parse_module().parse(stream).into_output_errors();

    for err in errs {
        println!("{:?}", err);
    }

    // Parsing should pass
    assert!(out.is_some());

    let out = out.unwrap();
    assert!(out.items.len() == 1);

    let finalf = match &out.items[0] {
        Declaration::Function(cl_func_dec) => cl_func_dec,
        _ => unreachable!(),
    };

    assert_eq!(finalf.airity(), 2);
    if let Type::Func { inputs, output, .. } = finalf.fntype() {
        println!("{:?}", inputs);
        assert!(inputs[0].is_err());
        assert!(inputs[1].is_ok());
        assert!(output.is_ok()); // Automatically get unit type
    }
}
