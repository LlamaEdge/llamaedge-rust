//! Parameters for the chat completion API.

use endpoints::chat::{ChatResponseFormat, Tool, ToolChoice};

/// Parameters for the chat completion API.
#[derive(Debug, Clone)]
pub struct ChatParams {
    /// The model to use for generating completions.
    pub model: Option<String>,
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to 1.0.
    pub temperature: Option<f64>,
    /// Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. The value should be between 0.0 and 1.0.
    ///
    /// Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
    ///
    /// We generally recommend altering this or temperature but not both.
    /// Defaults to 1.0.
    pub top_p: Option<f64>,
    /// How many chat completion choices to generate for each input message.
    /// Defaults to 1.
    pub n_choice: Option<u64>,
    /// A list of tokens at which to stop generation. If None, no stop tokens are used. Up to 4 sequences where the API will stop generating further tokens.
    /// Defaults to None
    pub stop: Option<Vec<String>>,
    /// **Deprecated** Use `max_completion_tokens` instead.
    ///
    /// The maximum number of tokens to generate. The value should be no less than 1.
    /// Defaults to 1024.
    pub max_tokens: Option<u64>,
    /// An upper bound for the number of tokens that can be generated for a completion. Defaults to -1, which means no upper bound.
    pub max_completion_tokens: Option<i32>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    /// Defaults to 0.0.
    pub presence_penalty: Option<f64>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    /// Defaults to 0.0.
    pub frequency_penalty: Option<f64>,
    /// A unique identifier representing your end-user.
    pub user: Option<String>,
    /// Format that the model must output
    pub response_format: Option<ChatResponseFormat>,
    /// A list of tools the model may call.
    ///
    /// Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.
    pub tools: Option<Vec<Tool>>,
    /// Controls which (if any) function is called by the model.
    pub tool_choice: Option<ToolChoice>,
}
impl Default for ChatParams {
    fn default() -> Self {
        Self {
            model: None,
            temperature: Some(1.0),
            top_p: Some(1.0),
            n_choice: Some(1),
            // stream: Some(false),
            // stream_options: None,
            stop: None,
            max_tokens: Some(1024),
            max_completion_tokens: Some(-1),
            presence_penalty: Some(0.0),
            frequency_penalty: Some(0.0),
            user: None,
            response_format: None,
            tools: None,
            tool_choice: None,
        }
    }
}
