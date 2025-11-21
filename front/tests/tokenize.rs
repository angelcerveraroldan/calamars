use front::syntax::token::Token;

#[test]
fn tokenize_expr_should_ignore_spaces() {
    let pairs = [("1 + 1", "1+1")];
    for (a, b) in pairs {
        let atks = Token::tokenize_line(a);
        let btks = Token::tokenize_line(b);
        assert_eq!(
            atks, btks,
            "Parsing shuold be the same ragardless of spacing"
        );
    }
}
