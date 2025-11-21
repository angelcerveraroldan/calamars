use calamars_core::ids::FileId;
use front::syntax::{
    ast::{Declaration, Expression, Item},
    token::Token,
};

#[test]
fn parse_fn_without_spaces() {
    let source = "def yyy() = 2+1";
    let tokens = Token::tokens_spanned_stream(source);

    let mut parser = front::syntax::parser::CalamarsParser::new(FileId::from(0), tokens);
    let item = parser.parse_item();

    assert!(
        parser.diag().is_empty(),
        "There shuold be no errors parsing"
    );

    let f = match item {
        Item::Declaration(Declaration::Function(fd)) => fd,
        _ => panic!("We should have parsed a function"),
    };

    let body = f.body();

    assert!(
        matches!(body, Expression::BinaryOp(_)),
        "Expression assigned should be a binary op"
    );
}

#[test]
fn parse_fn_with_spaces() {
    let source = "def yyy() = 2 + 1";
    let tokens = Token::tokens_spanned_stream(source);

    let mut parser = front::syntax::parser::CalamarsParser::new(FileId::from(0), tokens);
    let item = parser.parse_item();

    assert!(parser.is_finished(), "we shuold have consumed every token");
    assert!(
        parser.diag().is_empty(),
        "There should be no errors parsing"
    );

    let f = match item {
        Item::Declaration(Declaration::Function(fd)) => fd,
        _ => panic!("We should have parsed a function"),
    };

    let body = f.body();

    assert!(
        matches!(body, Expression::BinaryOp(_)),
        "Expression assigned should be a binary op"
    );
}
