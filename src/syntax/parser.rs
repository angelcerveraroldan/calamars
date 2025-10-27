use crate::{
    source::FileId,
    syntax::{
        ast::{self, FuncCall},
        errors::ParsingError,
        span::Span,
        token::Token,
    },
};

fn uni_span(s: &Span) -> Span {
    Span::from(s.start..s.start)
}

const UNARY_BP: usize = 90;
const FUN_CALL: usize = 80;

// Precedence information
#[rustfmt::kip]
const fn precedence(b: &Token) -> Option<(usize, usize)> {
    match b {
        Token::Pow => Some((70, 70)),
        Token::Star | Token::Slash => Some((60, 61)),
        Token::Plus | Token::Minus | Token::Concat => Some((50, 51)),
        Token::GreaterEqual | Token::LessEqual => Some((40, 41)),
        Token::EqualEqual | Token::NotEqual => Some((35, 36)),
        Token::And => Some((30, 31)),
        Token::Xor => Some((25, 26)),
        Token::Or => Some((20, 21)),
        _ => None,
    }
}

const fn binary_op_from_token(token: &Token) -> Option<ast::BinaryOperator> {
    match token {
        Token::Xor => Some(ast::BinaryOperator::Xor),
        Token::EqualEqual => Some(ast::BinaryOperator::EqEq),
        Token::NotEqual => Some(ast::BinaryOperator::NotEqual),
        Token::LessEqual => Some(ast::BinaryOperator::Leq),
        Token::GreaterEqual => Some(ast::BinaryOperator::Geq),
        Token::Concat => Some(ast::BinaryOperator::Concat),
        Token::Plus => Some(ast::BinaryOperator::Add),
        Token::Minus => Some(ast::BinaryOperator::Sub),
        Token::Star => Some(ast::BinaryOperator::Times),
        Token::Pow => Some(ast::BinaryOperator::Pow),
        Token::Slash => Some(ast::BinaryOperator::Div),
        Token::Less => Some(ast::BinaryOperator::Less),
        Token::Greater => Some(ast::BinaryOperator::Greater),
        _ => None,
    }
}

pub struct CalamarsParser {
    /// Id for the file for which this parser is responsible
    fileid: FileId,
    // Input tokens
    tokens: Vec<(Token, Span)>,
    /// Keep a set of errors that need to be reported
    diag: Vec<ParsingError>,
    curr_index: usize,
}

impl CalamarsParser {
    const ERR_IDENT: &'static str = "<err_ident>";

    /// This function guarantees:
    /// - Tokens is never empty
    pub fn new(fileid: FileId, tokens: Vec<(Token, logos::Span)>) -> Self {
        let mut tokens = tokens
            .into_iter()
            .map(|(t, s)| (t, Span::from(s)))
            .collect::<Vec<_>>();

        // If it does not end on an end of file token, then we add it
        if !matches!(tokens.last().map(|x| &x.0), Some(Token::EOF)) {
            let end = tokens.last().map(|(_, s)| s.end).unwrap_or_default();
            tokens.push((Token::EOF, Span::from(end..end)));
        }

        Self {
            fileid,
            tokens,
            diag: vec![],
            curr_index: 0,
        }
    }

    fn insert_err(&mut self, err: ParsingError) {
        self.diag.push(err);
    }

    fn expect_err(&mut self, expect: impl Into<String>) {
        self.insert_err(ParsingError::Expected {
            expected: expect.into(),
            span: self.zero_width_here(),
        });
    }

    /// Get the index of the EOF token
    fn eof_index(&self) -> usize {
        self.tokens.len().saturating_sub(1)
    }

    fn at_end(&self) -> bool {
        self.n_index() == self.eof_index()
    }

    fn zero_width_here(&self) -> Span {
        let span = self.next_ref().1;
        Span::from(span.start..span.start)
    }

    fn last_consumed_span(&self) -> Span {
        self.tokens[self.n_index().saturating_sub(1)].1
    }

    /// Get the index of the next non-consumed token
    ///
    /// We are guaranteed that this insdex will be in range for tokens indexing
    fn n_index(&self) -> usize {
        self.eof_index().min(self.curr_index)
    }

    /// Increase the current index by one
    fn advance_one(&mut self) {
        self.curr_index += 1;
    }

    /// Get the next token by reference
    fn next_token_ref(&self) -> &Token {
        &self.tokens[self.n_index()].0
    }

    fn next_ref(&self) -> &(Token, Span) {
        &self.tokens[self.n_index()]
    }

    fn next_if_matches(&self, f: impl FnOnce(&Token) -> bool) -> Option<&(Token, Span)> {
        if self.next_matches(f) {
            Some(self.next_ref())
        } else {
            None
        }
    }

    /// Check if the next token meets some predicate
    fn next_matches(&self, f: impl FnOnce(&Token) -> bool) -> bool {
        let next_token = self.next_token_ref();
        f(next_token)
    }

    /// Get the next n tokens
    fn peek_n(&self, n: usize) -> Box<[&Token]> {
        let next = self.n_index();
        let range = next..n.min(next);
        self.tokens[range]
            .iter()
            .map(|(a, _)| a)
            .collect::<Box<[&Token]>>()
    }

    fn checkpoint(&self) -> (usize, usize) {
        (self.curr_index, self.diag.len())
    }

    fn rollback(&mut self, (index, diagl): (usize, usize)) {
        self.curr_index = index;
        self.diag.truncate(diagl);
    }

    /// Keep skipping tokens until some predicate is met.
    ///
    /// This can be used for error recovery.
    fn skip_until<Ctx>(&mut self, mut init: Ctx, until: impl Fn(&mut Ctx, &Token) -> bool) {
        loop {
            let next_token = self.next_token_ref();
            if until(&mut init, next_token) || self.at_end() {
                return;
            }

            self.advance_one();
        }
    }

    fn parse_identifier(&mut self) -> ast::Ident {
        match self.next_ref().clone() {
            (Token::Ident(id), span) => {
                self.advance_one();
                ast::Ident::new(id, span)
            }
            (_, span) => {
                let unit_span = Span::from(span.start..span.start);
                self.insert_err(ParsingError::Expected {
                    expected: "identifier".to_string(),
                    span: unit_span,
                });
                // For now, usng <err_ident> is ok, since we dont allow idents to start with '<' in
                // the tokenizer. This is not great, but it will be fixed when we implement string
                // interning.
                ast::Ident::new(Self::ERR_IDENT.into(), unit_span)
            }
        }
    }

    /// Try to parse a literal.
    fn try_lit(&self) -> Option<ast::Literal> {
        let (tk, span) = self.next_ref();
        let lit_kind = match tk {
            Token::True => ast::LiteralKind::Boolean(true),
            Token::False => ast::LiteralKind::Boolean(false),
            Token::Int(i) => ast::LiteralKind::Integer(*i),
            Token::Float(f) => ast::LiteralKind::Real(*f),
            Token::Char(c) => ast::LiteralKind::Char(*c),
            Token::String(s) => ast::LiteralKind::String(s.clone()),
            _ => return None,
        };
        Some(ast::Literal::new(lit_kind, *span))
    }

    // Literals, name/path, bracketed expressions
    fn parse_primary(&mut self) -> ast::Expression {
        // A literal is the simplest case
        if let Some(literal) = self.try_lit() {
            self.advance_one();
            return ast::Expression::Literal(literal);
        }
        let (tk, sp) = self.next_ref().clone();
        match tk {
            // TODO: For now we just supprt idents to be one word
            Token::Ident(_) => ast::Expression::Identifier(self.parse_identifier()),
            Token::LParen => {
                self.advance_one();
                let expr = self.parse_expression();

                // Try to find closing paren
                // If we dont find it, we will artificially insert it
                if !self.next_matches(|t| *t == Token::RParen) {
                    self.insert_err(ParsingError::DelimeterNotClosed {
                        opening_loc: sp,
                        expected: ")",
                        at: self.zero_width_here(),
                    });
                }

                expr
            }
            // For no skip, not implemented
            Token::LBrace => {
                self.skip_until(1, |a, b| {
                    if matches!(b, Token::RBrace) {
                        *a -= 1;
                    }
                    if matches!(b, Token::LBrace) {
                        *a += 1;
                    }
                    *a == 0
                });
                ast::Expression::Error(self.zero_width_here())
            }
            _ => {
                let at = self.zero_width_here();
                self.insert_err(ParsingError::Expected {
                    expected: "expression".into(),
                    span: at,
                });
                ast::Expression::Error(at)
            }
        }
    }

    /// Parse comma separated arguments.
    ///
    /// To call this, you must ensure that the first token is LParen.
    fn parse_fn_arguments(&mut self) -> Vec<ast::Expression> {
        let mut arguments = vec![];
        self.advance_one();

        loop {
            // Handle the case where there are no arguments
            if self.next_matches(|tk| *tk == Token::RParen) {
                self.advance_one();
                return arguments;
            }

            arguments.push(self.parse_expression());

            if self.next_matches(|tk| *tk == Token::Comma) {
                self.advance_one(); // Consume the comma
            } else {
                break;
            }
        }

        if self.next_matches(|tk| *tk == Token::RParen) {
            self.advance_one();
        } else {
            self.expect_err(")");
        }

        arguments
    }

    /// Given some expression, apply postfix operations to it.
    ///
    /// This will handle things such as `f(x)`. We will apply `(x)` to `f`.
    fn parse_postfix_operation(&mut self, mut expr: ast::Expression) -> ast::Expression {
        loop {
            let (tk, sp) = self.next_ref().clone();
            match tk {
                // Apply parameters to input
                Token::LParen => {
                    let args = self.parse_fn_arguments();
                    let span_end = self.last_consumed_span();
                    let span = Span::from(sp.start..span_end.end);
                    expr = ast::Expression::FunctionCall(FuncCall::new(expr, args, span));
                }
                // TODO: Index not yet supported
                Token::LBracket => {
                    self.skip_until(1, |a, b| {
                        if matches!(b, Token::RBracket) {
                            *a -= 1;
                        }
                        if matches!(b, Token::LBracket) {
                            *a += 1;
                        }
                        *a == 0
                    });
                }
                _ => break,
            }
        }
        expr
    }

    fn parse_prefix(&mut self) -> ast::Expression {
        if self.next_matches(|t| *t == Token::Minus) {
            let unary_span = self.next_ref().1;
            self.advance_one();
            let expression = self.parse_expression_with_binding_power(UNARY_BP);
            let span = Span::from(unary_span.start..expression.span().end);
            let unary = ast::UnaryOp::new(ast::UnaryOperator::Neg, expression.into(), span);
            ast::Expression::UnaryOp(unary)
        } else {
            self.parse_primary()
        }
    }

    /// Pratt parsing for binary expressions
    ///
    /// `Expression <symbol> Expression` where <symbol> can be one of the following:
    /// <, <=, >, >=, |>, and, or, xor, *, ^, /, +, -
    fn parse_expression_with_binding_power(&mut self, binding_power: usize) -> ast::Expression {
        let mut lhs = self.parse_prefix();
        lhs = self.parse_postfix_operation(lhs);
        loop {
            let token = self.next_token_ref();
            let operator = precedence(token);
            if operator.is_none() {
                break;
            }

            let (lbp, rbp) = operator.unwrap();
            let operator = binary_op_from_token(token).unwrap();
            if lbp < binding_power {
                break;
            }

            // Consume the operator
            self.advance_one();
            let rhs = self.parse_expression_with_binding_power(rbp);
            let rhs = self.parse_postfix_operation(rhs);
            let span = Span::from(lhs.span().start..rhs.span().end);
            let binop = ast::BinaryOp::new(operator, lhs.into(), rhs.into(), span);
            lhs = ast::Expression::BinaryOp(binop);
        }
        lhs
    }

    fn parse_expression(&mut self) -> ast::Expression {
        self.parse_expression_with_binding_power(0)
    }

    /// Parse types
    ///
    /// Possible types currently implemented are:
    /// - Path (something like `String`, or `foo.Bar`)
    /// - Function (`Type -> Type`)
    /// - Array (`[ Type ]`)
    fn parse_type(&mut self) -> ast::Type {
        todo!()
    }

    /// Parse bindings such as
    ///
    /// ```cm
    /// var x: Int = expression;
    /// val y: Int = expression;
    /// ```
    fn parse_binding(&mut self) -> ast::Binding {
        todo!()
    }

    pub fn parse_file(&mut self) -> ast::Module {
        todo!()
    }

    pub fn diag(&self) -> &[ParsingError] {
        &self.diag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(start: usize, end: usize) -> logos::Span {
        logos::Span { start, end }
    }

    fn toks(pairs: &[(Token, (usize, usize))]) -> Vec<(Token, logos::Span)> {
        pairs
            .iter()
            .map(|(t, (s, e))| (t.clone(), sp(*s, *e)))
            .collect()
    }

    fn parse_expr_from_tokens(
        tokens: Vec<(Token, logos::Span)>,
    ) -> (CalamarsParser, ast::Expression) {
        let file = FileId(0);
        let mut p = CalamarsParser::new(file, tokens);
        let expr = p.parse_expression();
        (p, expr)
    }

    #[test]
    fn expr_precedence_mul_over_add() {
        // 1 + 2 * 3
        let tokens = toks(&[
            (Token::Int(1), (0, 1)),
            (Token::Plus, (2, 3)),
            (Token::Int(2), (4, 5)),
            (Token::Star, (6, 7)),
            (Token::Int(3), (8, 9)),
            (Token::EOF, (9, 9)),
        ]);
        let (p, e) = parse_expr_from_tokens(tokens);

        // 2 * 3
        let mult = ast::Expression::BinaryOp(ast::BinaryOp::new(
            ast::BinaryOperator::Times,
            Box::new(ast::Expression::Literal(ast::Literal::new(
                ast::LiteralKind::Integer(2),
                Span::from(4..5),
            ))),
            Box::new(ast::Expression::Literal(ast::Literal::new(
                ast::LiteralKind::Integer(3),
                Span::from(8..9),
            ))),
            Span::from(4..9),
        ));

        // 1 + mult
        let expected = ast::Expression::BinaryOp(ast::BinaryOp::new(
            ast::BinaryOperator::Add,
            Box::new(ast::Expression::Literal(ast::Literal::new(
                ast::LiteralKind::Integer(1),
                Span::from(0..1),
            ))),
            Box::new(mult),
            Span::from(0..9),
        ));

        assert!(p.diag.is_empty(), "should parse without diagnostics");
        assert!(e == expected);
    }

    #[test]
    fn expr_calls_chain() {
        // f(1)(2 + 3)
        let tokens = toks(&[
            (Token::Ident("f".into()), (0, 1)),
            (Token::LParen, (1, 2)),
            (Token::Int(1), (2, 3)),
            (Token::RParen, (3, 4)),
            (Token::LParen, (4, 5)),
            (Token::Int(2), (5, 6)),
            (Token::Plus, (7, 8)),
            (Token::Int(3), (9, 10)),
            (Token::RParen, (10, 11)),
            (Token::EOF, (11, 11)),
        ]);
        let (p, e) = parse_expr_from_tokens(tokens);

        assert!(matches!(e, ast::Expression::FunctionCall(_)));
        assert!(p.diag.is_empty(), "call chaining should parse cleanly");
    }

    #[test]
    fn expr_grouping_then_mul() {
        // (a + b) * c
        let tokens = toks(&[
            (Token::LParen, (0, 1)),
            (Token::Ident("a".into()), (1, 2)),
            (Token::Plus, (3, 4)),
            (Token::Ident("b".into()), (5, 6)),
            (Token::RParen, (6, 7)),
            (Token::Star, (8, 9)),
            (Token::Ident("c".into()), (10, 11)),
            (Token::EOF, (11, 11)),
        ]);
        let (p, e) = parse_expr_from_tokens(tokens);
        assert!(matches!(e, ast::Expression::BinaryOp(_)));
        match e {
            ast::Expression::BinaryOp(b) => {
                assert!(matches!(b.rhs().as_ref(), ast::Expression::Identifier(_)));
            }
            _ => panic!(),
        }
        assert!(
            p.diag.is_empty(),
            "grouping + precedence should parse without diagnostics"
        );
    }

    #[test]
    fn expr_missing_rparen_in_call_emits_diag() {
        // f(1
        let tokens = toks(&[
            (Token::Ident("f".into()), (0, 1)),
            (Token::LParen, (1, 2)),
            (Token::Int(1), (2, 3)),
            (Token::EOF, (3, 3)),
        ]);
        let (p, _e) = parse_expr_from_tokens(tokens);
        assert!(!p.diag.is_empty(), "should report a missing `)` diagnostic");
    }

    #[test]
    fn expr_missing_rhs_after_plus_emits_diag() {
        // 1 +
        let tokens = toks(&[
            (Token::Int(1), (0, 1)),
            (Token::Plus, (2, 3)),
            (Token::EOF, (3, 3)),
        ]);
        let (mut p, _e) = parse_expr_from_tokens(tokens);
        let _ = p.parse_expression();

        assert!(
            !p.diag.is_empty(),
            "should report an expected expression after `+`"
        );
    }
}
