use thiserror::Error;

/// Error types for the Llama Core library.
#[derive(Error, Debug)]
pub enum LlamaEdgeError {
    /// Errors in General operation.
    #[error("{0}")]
    Operation(String),
    /// Errors in URL parsing.
    #[error("Invalid URL: {0}")]
    UrlParse(#[from] url::ParseError),
    /// Errors in invalid argument.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}
