use chumsky::{error::Simple, span::SimpleSpan};

use crate::source::FileId;

pub struct FileSpan {
    start: usize,
    end: usize,
    file: FileId,
}

impl FileSpan {
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn file(&self) -> &FileId {
        &self.file
    }
}

impl From<(logos::Span, FileId)> for FileSpan {
    fn from((s, f): (logos::Span, FileId)) -> Self {
        Self {
            start: s.start,
            end: s.end,
            file: f,
        }
    }
}

impl From<SimpleSpan<usize, FileId>> for FileSpan {
    fn from(value: SimpleSpan<usize, FileId>) -> Self {
        FileSpan {
            start: value.start,
            end: value.end,
            file: value.context,
        }
    }
}

// For interpreting single files - May delete later
impl From<SimpleSpan> for FileSpan {
    fn from(value: SimpleSpan) -> Self {
        FileSpan {
            start: value.start,
            end: value.end,
            file: FileId(0),
        }
    }
}

pub type Span = SimpleSpan;
