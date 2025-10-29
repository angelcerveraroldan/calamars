use ariadne::{Color, Label, Report, ReportKind, Source};
use std::ops::Range;

use crate::syntax::span::Span;

pub fn label_from(
    file_name: &String,
    span: Span,
    message: impl Into<String>,
    color: Option<Color>,
) -> Label<(&String, Range<usize>)> {
    let mut label = Label::new((file_name, span.into_range())).with_message(message.into());
    if let Some(color) = color {
        label = label.with_color(color);
    }
    label
}

pub trait PrettyError {
    fn labels<'a>(&'a self, file_name: &'a String) -> Vec<Label<(&'a String, Range<usize>)>>;
    fn notes(&self) -> Option<String>;
    fn message(&self) -> &str;

    /// Given a semantic error, pretty print it with the source code
    fn log_error(&self, file_name: &String, source: &String) -> Result<(), std::io::Error> {
        let rep = Report::build(ReportKind::Error, (file_name, 12..12))
            .with_message(self.message().to_string());

        let mut rep_labelled = self
            .labels(file_name)
            .into_iter()
            .fold(rep, |rep, label| rep.with_label(label));

        if let Some(note) = self.notes() {
            rep_labelled = rep_labelled.with_note(note);
        };

        let fin = rep_labelled.finish();
        fin.print((file_name, Source::from(source)))
    }
}
