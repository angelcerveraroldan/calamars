use crate::{
    source::FileId,
    syntax::{ast, errors::ParsingError, span::Span, token::Token},
};

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

    /// Get the index of the EOF token
    fn eof_index(&self) -> usize {
        self.tokens.len().saturating_sub(1)
    }

    fn at_end(&self) -> bool {
        self.n_index() == self.eof_index()
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
    fn skip_to(&mut self, until: fn(&Token) -> bool) {
        loop {
            if self.next_matches(until) || self.at_end() {
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

    /// Parse types
    fn parse_type_path(&mut self) -> ast::Type {
        todo!()
    }

    /// Parse bindings such as
    ///
    /// ```
    /// var x: Int = expression;
    /// val y: Int = expression;
    /// ```
    fn parse_binding(&mut self) -> ast::Binding {
        todo!()
    }
}

#[cfg(test)]
mod test_parser {
    use crate::syntax::token::Token;

    fn identifier() -> Token {
        Token::Ident("Hello".into())
    }

    #[test]
    fn parse_ident() {}
}
