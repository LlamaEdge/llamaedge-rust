//! Parameters for the chat completion API.

#[cfg(feature = "audio")]
use endpoints::audio::transcription::TimestampGranularity;
use endpoints::chat::{ChatResponseFormat, Tool, ToolChoice};
#[cfg(feature = "image")]
use endpoints::{
    files::FileObject,
    images::{SamplingMethod, Scheduler},
};
#[cfg(feature = "image")]
use std::path::PathBuf;

#[cfg(feature = "image")]
pub type ImageResponseFormat = endpoints::images::ResponseFormat;

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

/// Parameters for the RAG chat completion API.
#[cfg(feature = "rag")]
#[derive(Debug, Clone)]
pub struct RagChatParams {
    /// The model to use for generating completions.
    pub model: Option<String>,
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to 0.8.
    pub temperature: f64,
    /// Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. The value should be between 0.0 and 1.0.
    ///
    /// Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
    ///
    /// We generally recommend altering this or temperature but not both.
    /// Defaults to 0.9. To disable top-p sampling, set it to 1.0.
    pub top_p: f64,
    /// How many chat completion choices to generate for each input message.
    /// Defaults to 1.
    pub n_choice: u64,
    /// A list of tokens at which to stop generation. If None, no stop tokens are used. Up to 4 sequences where the API will stop generating further tokens.
    /// Defaults to None
    pub stop: Option<Vec<String>>,
    /// **Deprecated** Use `max_completion_tokens` instead.
    ///
    /// The maximum number of tokens to generate. The value should be no less than 1.
    /// Defaults to 1024.
    pub max_tokens: u64,
    /// An upper bound for the number of tokens that can be generated for a completion. Defaults to -1, which means no upper bound.
    pub max_completion_tokens: i32,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    /// Defaults to 0.0.
    pub presence_penalty: f64,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    /// Defaults to 0.0.
    pub frequency_penalty: f64,
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
    /// Number of user messages to use for context retrieval. Defaults to 1.
    pub context_window: u64,
    /// The configuration for the VectorDB server.
    pub vdb_config: Option<RagVdbConfig>,
}
#[cfg(feature = "rag")]
impl Default for RagChatParams {
    fn default() -> Self {
        Self {
            model: None,
            temperature: 0.8,
            top_p: 0.9,
            n_choice: 1,
            stop: None,
            max_tokens: 1024,
            max_completion_tokens: -1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            context_window: 1,
            vdb_config: None,
        }
    }
}

/// The configuration for the VectorDB server.
#[cfg(feature = "rag")]
#[derive(Debug, Clone)]
pub struct RagVdbConfig {
    /// The URL of the VectorDB server.
    pub server_url: String,
    /// The names of the collections in VectorDB.
    pub collection_name: Vec<String>,
    /// Max number of retrieved results. The number of the values must be the same as the number of `collection_name`.
    pub limit: Vec<u64>,
    /// The score threshold for the retrieved results. The number of the values must be the same as the number of `collection_name`.
    pub score_threshold: Vec<f32>,
    /// The API key for the VectorDB server.
    pub api_key: Option<String>,
}

/// Parameters for the transcription API.
#[cfg(feature = "audio")]
#[derive(Debug, Clone)]
pub struct TranscriptionParams {
    /// ID of the model to use.
    pub model: Option<String>,
    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
    pub prompt: Option<String>,
    /// The format of the transcript output, in one of these options: `json`, `text`, `srt`, `verbose_json`, or `vtt`.
    pub response_format: String,
    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically increase the temperature until certain thresholds are hit. Defaults to 0.0.
    pub temperature: f64,
    /// The timestamp granularities to populate for this transcription.
    /// `response_format` must be set `verbose_json` to use timestamp granularities. Either or both of these options are supported: `word`, or `segment`.
    pub timestamp_granularities: Option<Vec<TimestampGranularity>>,

    /// Automatically detect the spoken language in the provided audio input. Defaults to false.
    pub detect_language: bool,
    /// Time offset in milliseconds. Defaults to 0.
    pub offset_time: u64,
    /// Length of audio (in seconds) to be processed starting from the point defined by the `offset_time` field (or from the beginning by default). Defaults to 0.
    pub duration: u64,
    /// Maximum amount of text context (in tokens) that the model uses when processing long audio inputs incrementally. Defaults to -1.
    pub max_context: i32,
    /// Maximum number of tokens that the model can generate in a single transcription segment (or chunk). Defaults to 0.
    pub max_len: u64,
    /// Split audio chunks on word rather than on token. Defaults to false.
    pub split_on_word: bool,
    /// Use the new computation context. Defaults to false.
    pub use_new_context: bool,
}
#[cfg(feature = "audio")]
impl Default for TranscriptionParams {
    fn default() -> Self {
        Self {
            model: None,
            prompt: None,
            response_format: "json".to_string(),
            temperature: 0.0,
            timestamp_granularities: Some(vec![TimestampGranularity::Segment]),
            detect_language: false,
            offset_time: 0,
            duration: 0,
            max_context: -1,
            max_len: 0,
            split_on_word: false,
            use_new_context: false,
        }
    }
}

/// Parameters for the translation API.
#[cfg(feature = "audio")]
#[derive(Debug, Clone)]
pub struct TranslationParams {
    /// ID of the model to use.
    pub model: Option<String>,
    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should be in English.
    pub prompt: Option<String>,
    /// The format of the transcript output, in one of these options: `json`, `text`, `srt`, `verbose_json`, or `vtt`. Defaults to `json`.
    pub response_format: String,
    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically increase the temperature until certain thresholds are hit. Defaults to 0.0.
    pub temperature: f64,
    /// automatically detect the spoken language in the provided audio input. Defaults to false.
    pub detect_language: bool,
    /// Time offset in milliseconds. Defaults to 0.
    pub offset_time: u64,
    /// Length of audio (in seconds) to be processed starting from the point defined by the `offset_time` field (or from the beginning by default). Defaults to 0.
    pub duration: u64,
    /// Maximum amount of text context (in tokens) that the model uses when processing long audio inputs incrementally. Defaults to -1.
    pub max_context: i32,
    /// Maximum number of tokens that the model can generate in a single transcription segment (or chunk). Defaults to 0.
    pub max_len: u64,
    /// Split audio chunks on word rather than on token. Defaults to false.
    pub split_on_word: bool,
    /// Use the new computation context. Defaults to false.
    pub use_new_context: bool,
}
#[cfg(feature = "audio")]
impl Default for TranslationParams {
    fn default() -> Self {
        Self {
            model: None,
            prompt: None,
            response_format: "json".to_string(),
            temperature: 0.0,
            detect_language: false,
            offset_time: 0,
            duration: 0,
            max_context: -1,
            max_len: 0,
            split_on_word: false,
            use_new_context: false,
        }
    }
}

/// Parameters for the embeddings API.
#[derive(Debug, Clone)]
pub struct EmbeddingsParams {
    /// ID of the model to use.
    pub model: Option<String>,
    /// The format to return the embeddings in. Can be either float or base64.
    /// Defaults to float.
    pub encoding_format: String,
    /// A unique identifier representing your end-user.
    pub user: Option<String>,
    /// The URL of the VectorDB server.
    #[cfg(feature = "rag")]
    pub vdb_server_url: Option<String>,
    /// The name of the collection in VectorDB.
    #[cfg(feature = "rag")]
    pub vdb_collection_name: Option<String>,
    /// The API key for the VectorDB server.
    #[cfg(feature = "rag")]
    pub vdb_api_key: Option<String>,
}
impl Default for EmbeddingsParams {
    fn default() -> Self {
        Self {
            model: None,
            encoding_format: "float".to_string(),
            user: None,
            #[cfg(feature = "rag")]
            vdb_server_url: None,
            #[cfg(feature = "rag")]
            vdb_collection_name: None,
            #[cfg(feature = "rag")]
            vdb_api_key: None,
        }
    }
}

/// Parameters for the image generation API.
#[cfg(feature = "image")]
#[derive(Debug, Clone)]
pub struct ImageCreateParams {
    /// Negative prompt for the image generation.
    pub negative_prompt: Option<String>,
    /// Name of the model to use for image generation.
    pub model: String,
    /// Number of images to generate. Defaults to `1`.
    pub n: u64,
    /// The format in which the generated images are returned. Must be one of `url` or `b64_json`. Defaults to `Url`.
    pub response_format: ImageResponseFormat,
    /// A unique identifier representing your end-user, which can help monitor and detect abuse.
    pub user: Option<String>,
    /// Unconditional guidance scale. Defaults to `7.0`.
    pub cfg_scale: f32,
    /// Sampling method. Defaults to "euler_a".
    pub sample_method: SamplingMethod,
    /// Number of sample steps. Defaults to `20`.
    pub steps: usize,
    /// Image height, in pixel space. Defaults to `512`.
    pub height: usize,
    /// Image width, in pixel space. Defaults to `512`.
    pub width: usize,
    /// Strength to apply Control Net. Defaults to `0.9`.
    pub control_strength: f32,
    /// The image to control the generation.
    pub control_image: Option<FileObject>,
    /// RNG seed. Negative value means to use random seed. Defaults to `42`.
    pub seed: i32,
    /// Strength for noising/unnoising. Defaults to `0.75`.
    pub strength: f32,
    /// Denoiser sigma scheduler. Possible values are `discrete`, `karras`, `exponential`, `ays`, `gits`. Defaults to `discrete`.
    pub scheduler: Scheduler,
    /// Apply canny preprocessor. Defaults to `false`.
    pub apply_canny_preprocessor: bool,
    /// Strength for keeping input identity. Defaults to `0.2`.
    pub style_ratio: f32,
}
#[cfg(feature = "image")]
impl Default for ImageCreateParams {
    fn default() -> Self {
        Self {
            negative_prompt: None,
            model: "".to_string(),
            n: 1,
            response_format: ImageResponseFormat::Url,
            user: None,
            cfg_scale: 7.0,
            sample_method: SamplingMethod::EulerA,
            steps: 20,
            height: 512,
            width: 512,
            control_strength: 0.9,
            control_image: None,
            seed: 42,
            strength: 0.75,
            scheduler: Scheduler::Discrete,
            apply_canny_preprocessor: false,
            style_ratio: 0.2,
        }
    }
}

/// Parameters for the image edit API.
#[cfg(feature = "image")]
#[derive(Debug, Clone)]
pub struct ImageEditParams {
    /// Negative prompt for the image generation.
    pub negative_prompt: Option<String>,
    /// An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited. Must have the same dimensions as the input image.
    pub mask: Option<PathBuf>,
    /// The model to use for image generation.
    pub model: String,
    /// The number of images to generate. Defaults to `1`.
    pub n: u64,
    /// The format in which the generated images are returned. Must be one of `url` or `b64_json`. Defaults to `url`.
    pub response_format: ImageResponseFormat,
    /// A unique identifier representing your end-user, which can help monitor and detect abuse.
    pub user: Option<String>,
    /// Unconditional guidance scale. Defaults to `7.0`.
    pub cfg_scale: f32,
    /// Sampling method. Defaults to "euler_a".
    pub sample_method: SamplingMethod,
    /// Number of sample steps. Defaults to `20`.
    pub steps: usize,
    /// Image height, in pixel space. Defaults to `512`.
    pub height: usize,
    /// Image width, in pixel space. Defaults to `512`.
    pub width: usize,
    /// strength to apply Control Net. Defaults to `0.9`.
    pub control_strength: f32,
    /// The image to control the generation.
    pub control_image: Option<PathBuf>,
    /// RNG seed. Negative value means to use random seed. Defaults to `42`.
    pub seed: i32,
    /// Strength for noising/unnoising. Defaults to `0.75`.
    pub strength: f32,
    /// Denoiser sigma scheduler. Possible values are `discrete`, `karras`, `exponential`, `ays`, `gits`. Defaults to `discrete`.
    pub scheduler: Scheduler,
    /// Apply canny preprocessor. Defaults to `false`.
    pub apply_canny_preprocessor: bool,
    /// Strength for keeping input identity. Defaults to `0.2`.
    pub style_ratio: f32,
}
#[cfg(feature = "image")]
impl Default for ImageEditParams {
    fn default() -> Self {
        Self {
            negative_prompt: None,
            mask: None,
            model: "".to_string(),
            n: 1,
            response_format: ImageResponseFormat::Url,
            user: None,
            cfg_scale: 7.0,
            sample_method: SamplingMethod::EulerA,
            steps: 20,
            height: 512,
            width: 512,
            control_strength: 0.9,
            control_image: None,
            seed: 42,
            strength: 0.75,
            scheduler: Scheduler::Discrete,
            apply_canny_preprocessor: false,
            style_ratio: 0.2,
        }
    }
}
