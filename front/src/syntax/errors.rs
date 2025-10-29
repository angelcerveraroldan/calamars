use ariadne::Color;

use crate::{
    errors::{PrettyError, label_from},
    syntax::span::Span,
};

#[derive(Debug, Clone)]
pub enum ParsingError {
    Expected {
        expected: String,
        span: Span,
    },
    DelimeterNotClosed {
        expected: &'static str,
        at: Span,
        opening_loc: Span,
    },
}

impl PrettyError for ParsingError {
    fn labels<'a>(
        &'a self,
        file_name: &'a String,
    ) -> Vec<ariadne::Label<(&'a String, std::ops::Range<usize>)>> {
        match self {
            ParsingError::Expected { expected, span } => {
                vec![label_from(
                    file_name,
                    *span,
                    format!("Expected to find: {expected}"),
                    Some(Color::Red),
                )]
            }
            ParsingError::DelimeterNotClosed {
                expected,
                at,
                opening_loc,
            } => vec![
                label_from(
                    file_name,
                    *opening_loc,
                    "Opening delimeter here",
                    Some(Color::Green),
                ),
                label_from(file_name, *at, "No closing delimeter", Some(Color::Red)),
            ],
        }
    }

    fn notes(&self) -> Option<String> {
        match self {
            ParsingError::Expected { expected, span } => None,
            ParsingError::DelimeterNotClosed { .. } => None,
        }
    }

    fn message(&self) -> &str {
        match self {
            ParsingError::Expected { .. } => "Did not find expected token",
            ParsingError::DelimeterNotClosed { .. } => "Delimeter not closed",
        }
    }

    fn log_error(&self, file_name: &String, source: &String) -> Result<(), std::io::Error> {
        let rep = ariadne::Report::build(ariadne::ReportKind::Error, (file_name, 12..12))
            .with_message(self.message().to_string());

        let mut rep_labelled = self
            .labels(file_name)
            .into_iter()
            .fold(rep, |rep, label| rep.with_label(label));

        if let Some(note) = self.notes() {
            rep_labelled = rep_labelled.with_note(note);
        };

        let fin = rep_labelled.finish();
        fin.print((file_name, ariadne::Source::from(source)))
    }
}
