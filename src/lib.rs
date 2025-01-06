//! The `llamaedge` crate provides convenient access to the LlamaEdge REST API from any
//! Rust application. The crate includes type definitions for all request params and
//! response fields, and offers both synchronous and asynchronous clients.
//!
//! This project is still in the early stages of development. The API is not stable and may change.
//!
//! ## Usage
//!
//! Add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! llamaedge = "0.0.1"
//! endpoints = "0.23.0"
//! tokio = { version = "1.42.0", features = ["full"] }
//! ```
//!
//! Then, you can use the following code to send a chat completion request:

//! ```rust
//! use endpoints::chat::{
//!     ChatCompletionRequestMessage, ChatCompletionSystemMessage, ChatCompletionUserMessage,
//!    ChatCompletionUserMessageContent,
//! };
//! use llamaedge::{params::ChatParams, Client};

//! #[tokio::main]
//! async fn main() {
//!     const SERVER_BASE_URL: &str = "http://localhost:8080";
//!
//!     // Create a client
//!     let client = Client::new(SERVER_BASE_URL).unwrap();
//!
//!     // create messages
//!     let mut messages = Vec::new();
//!     let system_message = ChatCompletionRequestMessage::System(ChatCompletionSystemMessage::new(
//!         "You are a helpful assistant. Answer questions as concisely and accurately as possible.",
//!         None,
//!     ));
//!     messages.push(system_message);
//!     let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
//!         ChatCompletionUserMessageContent::Text("What is the capital of France?".to_string()),
//!         None,
//!     ));
//!     messages.push(user_message);
//!
//!     // send chat completion request
//!     if let Ok(generation) = client.chat(&messages[..], &ChatParams::default()).await {
//!         println!("assistant:{}", generation);
//!     }
//! }
//! ```
//!
//! **Note:** To run the example, LlamaEdge API server should be deployed and running on your local machine. Refer to [Quick Start](https://github.com/LlamaEdge/LlamaEdge?tab=readme-ov-file#quick-start) for more details on how to deploy and run the server.

pub mod error;
pub mod params;

use endpoints::{
    audio::{transcription::TranscriptionObject, translation::TranslationObject},
    chat::{
        ChatCompletionObject, ChatCompletionRequest, ChatCompletionRequestMessage, StreamOptions,
    },
    embeddings::{EmbeddingRequest, EmbeddingsResponse, InputText},
    files::FileObject,
    images::{ImageCreateRequestBuilder, ImageObject, ListImagesResponse},
    models::{ListModelsResponse, Model},
};
use error::LlamaEdgeError;
use futures::{stream::TryStream, StreamExt};
use params::{
    ChatParams, EmbeddingsParams, ImageCreateParams, ImageEditParams, TranscriptionParams,
    TranslationParams,
};
use reqwest::multipart;
use std::path::Path;
use url::Url;

/// Client for the LlamaEdge API.
pub struct Client {
    server_base_url: Url,
}
impl Client {
    /// Create a new client.
    ///
    /// # Arguments
    ///
    /// * `server_base_url` - The base URL of the LlamaEdge API server.
    ///
    /// # Returns
    ///
    /// A `Result` containing the client or an error.
    pub fn new(server_base_url: impl AsRef<str>) -> Result<Self, LlamaEdgeError> {
        let url_str = server_base_url.as_ref().trim_end_matches('/');
        match Url::parse(url_str) {
            Ok(url) => Ok(Self {
                server_base_url: url,
            }),
            Err(e) => {
                return Err(LlamaEdgeError::UrlParse(e));
            }
        }
    }

    /// Get the server base URL.
    ///
    /// # Returns
    ///
    /// A reference to the server base URL.
    pub fn server_base_url(&self) -> &Url {
        &self.server_base_url
    }

    /// Send a chat completion request.
    ///
    /// # Arguments
    ///
    /// * `chat_history` - The chat history including the latest user message.
    ///
    /// * `params` - The parameters for the chat completion.
    ///
    /// # Returns
    ///
    /// A `Result` containing the chat completion or an error.
    pub async fn chat(
        &self,
        chat_history: &[ChatCompletionRequestMessage],
        params: &ChatParams,
    ) -> Result<String, LlamaEdgeError> {
        if chat_history.is_empty() {
            return Err(LlamaEdgeError::InvalidArgument(
                "chat_history cannot be empty".to_string(),
            ));
        }

        // create request for chat completion
        let request = ChatCompletionRequest {
            messages: chat_history.to_vec(),
            model: params.model.clone(),
            temperature: params.temperature,
            top_p: params.top_p,
            n_choice: params.n_choice,
            stop: params.stop.clone(),
            max_tokens: params.max_tokens,
            // max_completion_tokens: params.max_completion_tokens,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            user: params.user.clone(),
            response_format: params.response_format.clone(),
            tools: params.tools.clone(),
            tool_choice: params.tool_choice.clone(),
            ..Default::default()
        };

        let url = self.server_base_url.join("/v1/chat/completions")?;
        let response = reqwest::Client::new()
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        let response_body = response
            .json::<ChatCompletionObject>()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        match &response_body.choices[0].message.content {
            Some(content) => Ok(content.clone()),
            None => Ok("".to_string()),
        }
    }

    /// Send a chat completion request with streaming.
    ///
    /// # Arguments
    ///
    /// * `chat_history` - The chat history including the latest user message.
    ///
    /// * `params` - The parameters for the chat completion.
    ///
    /// # Returns
    ///
    /// A `Result` containing the chat completion stream or an error.
    pub async fn chat_stream(
        &self,
        chat_history: &[ChatCompletionRequestMessage],
        params: &ChatParams,
    ) -> Result<
        impl TryStream<Item = Result<String, LlamaEdgeError>, Error = LlamaEdgeError>,
        LlamaEdgeError,
    > {
        if chat_history.is_empty() {
            return Err(LlamaEdgeError::InvalidArgument(
                "chat_history cannot be empty".to_string(),
            ));
        }

        // create request for chat completion
        let request = ChatCompletionRequest {
            messages: chat_history.to_vec(),
            model: params.model.clone(),
            temperature: params.temperature,
            top_p: params.top_p,
            n_choice: params.n_choice,
            stop: params.stop.clone(),
            max_tokens: params.max_tokens,
            // max_completion_tokens: params.max_completion_tokens,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            user: params.user.clone(),
            response_format: params.response_format.clone(),
            tools: params.tools.clone(),
            tool_choice: params.tool_choice.clone(),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
            }),
            ..Default::default()
        };

        let url = self.server_base_url.join("/v1/chat/completions")?;
        let response = reqwest::Client::new()
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        let stream = response.bytes_stream().map(|r| match r {
            Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).to_string()),
            Err(e) => Err(LlamaEdgeError::Operation(e.to_string())),
        });

        Ok(stream)
    }

    /// Transcribe an audio file.
    ///
    /// # Arguments
    ///
    /// * `audio_file` - The audio file to transcribe.
    ///
    /// * `spoken_language` - The language of the audio file. The language should be in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format. For example, "en" for English, "zh" for Chinese, "ja" for Japanese, etc.
    ///
    /// * `params` - The parameters for the transcription.
    ///
    /// # Returns
    ///
    /// A `Result` containing the transcription object or an error.
    pub async fn transcribe(
        &self,
        audio_file: impl AsRef<Path>,
        spoken_language: impl AsRef<str>,
        params: TranscriptionParams,
    ) -> Result<TranscriptionObject, LlamaEdgeError> {
        let abs_file_path = if audio_file.as_ref().is_absolute() {
            audio_file.as_ref().to_path_buf()
        } else {
            std::env::current_dir().unwrap().join(audio_file.as_ref())
        };

        // check if the file exists
        if !abs_file_path.exists() {
            let error_message =
                format!("The audio file does not exist: {}", abs_file_path.display());

            return Err(LlamaEdgeError::InvalidArgument(error_message));
        }

        // get the filename
        let filename = abs_file_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        // get the file extension
        let file_extension = abs_file_path
            .extension()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let file = tokio::fs::read(abs_file_path).await.map_err(|e| {
            LlamaEdgeError::Operation(format!("Failed to read the audio file: {}", e))
        })?;

        let form = {
            let file_part = multipart::Part::bytes(file)
                .file_name(filename)
                .mime_str(&format!("audio/{}", file_extension))
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let language = if spoken_language.as_ref().is_empty() {
                "en".to_string()
            } else {
                spoken_language.as_ref().to_string()
            };
            let language_part = multipart::Part::text(language)
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let response_format_part = multipart::Part::text(params.response_format)
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let temperature_part = multipart::Part::text(params.temperature.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let detect_language_part = multipart::Part::text(params.detect_language.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let offset_time_part = multipart::Part::text(params.offset_time.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let duration_part = multipart::Part::text(params.duration.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let max_context_part = multipart::Part::text(params.max_context.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let max_len_part = multipart::Part::text(params.max_len.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let split_on_word_part = multipart::Part::text(params.split_on_word.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let use_new_context_part = multipart::Part::text(params.use_new_context.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let mut form = multipart::Form::new()
                .part("file", file_part)
                .part("language", language_part)
                .part("response_format", response_format_part)
                .part("temperature", temperature_part)
                .part("detect_language", detect_language_part)
                .part("offset_time", offset_time_part)
                .part("duration", duration_part)
                .part("max_context", max_context_part)
                .part("max_len", max_len_part)
                .part("split_on_word", split_on_word_part)
                .part("use_new_context", use_new_context_part);

            if let Some(model) = &params.model {
                let model_part = multipart::Part::text(model.clone())
                    .mime_str("text/plain")
                    .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;
                form = form.part("model", model_part);
            }

            if let Some(prompt) = &params.prompt {
                let prompt_part = multipart::Part::text(prompt.clone())
                    .mime_str("text/plain")
                    .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;
                form = form.part("prompt", prompt_part);
            }

            form
        };

        // send the transcription request
        let url = self.server_base_url.join("/v1/audio/transcriptions")?;
        let response = reqwest::Client::new()
            .post(url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        // get the transcription object
        let transcription_object = response
            .json::<TranscriptionObject>()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        Ok(transcription_object)
    }

    /// Translate an audio file.
    ///
    /// # Arguments
    ///
    /// * `audio_file` - The audio file to translate.
    ///
    /// * `spoken_language` - The language of the audio file. The language should be in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format. For example, "en" for English, "zh" for Chinese, "ja" for Japanese, etc.
    ///
    /// * `params` - The parameters for the translation.
    ///
    /// # Returns
    ///
    /// A `Result` containing the translation object or an error.
    pub async fn translate(
        &self,
        audio_file: impl AsRef<Path>,
        spoken_language: impl AsRef<str>,
        params: TranslationParams,
    ) -> Result<TranslationObject, LlamaEdgeError> {
        let abs_file_path = if audio_file.as_ref().is_absolute() {
            audio_file.as_ref().to_path_buf()
        } else {
            std::env::current_dir().unwrap().join(audio_file.as_ref())
        };

        // check if the file exists
        if !abs_file_path.exists() {
            let error_message =
                format!("The audio file does not exist: {}", abs_file_path.display());

            return Err(LlamaEdgeError::InvalidArgument(error_message));
        }

        // get the filename
        let filename = abs_file_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        // get the file extension
        let file_extension = abs_file_path
            .extension()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let file = tokio::fs::read(abs_file_path)
            .await
            .map_err(|e| LlamaEdgeError::Operation(format!("Failed to read audio file: {}", e)))?;

        let form = {
            let file_part = multipart::Part::bytes(file)
                .file_name(filename)
                .mime_str(&format!("audio/{}", file_extension))
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let response_format_part = multipart::Part::text(params.response_format)
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let language = if spoken_language.as_ref().is_empty() {
                "en".to_string()
            } else {
                spoken_language.as_ref().to_string()
            };
            let language_part = multipart::Part::text(language)
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let temperature_part = multipart::Part::text(params.temperature.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let detect_language_part = multipart::Part::text(params.detect_language.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let offset_time_part = multipart::Part::text(params.offset_time.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let duration_part = multipart::Part::text(params.duration.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let max_context_part = multipart::Part::text(params.max_context.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let max_len_part = multipart::Part::text(params.max_len.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let split_on_word_part = multipart::Part::text(params.split_on_word.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let use_new_context_part = multipart::Part::text(params.use_new_context.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let mut form = multipart::Form::new()
                .part("file", file_part)
                .part("response_format", response_format_part)
                .part("language", language_part)
                .part("temperature", temperature_part)
                .part("detect_language", detect_language_part)
                .part("offset_time", offset_time_part)
                .part("duration", duration_part)
                .part("max_context", max_context_part)
                .part("max_len", max_len_part)
                .part("split_on_word", split_on_word_part)
                .part("use_new_context", use_new_context_part);

            if let Some(model) = &params.model {
                let model_part = multipart::Part::text(model.clone())
                    .mime_str("text/plain")
                    .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;
                form = form.part("model", model_part);
            }

            if let Some(prompt) = &params.prompt {
                let prompt_part = multipart::Part::text(prompt.clone())
                    .mime_str("text/plain")
                    .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;
                form = form.part("prompt", prompt_part);
            }

            form
        };

        // send the transcription request
        let url = self.server_base_url.join("/v1/audio/translations")?;
        let response = reqwest::Client::new()
            .post(url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        // get the translation object
        let translation_object = response
            .json::<TranslationObject>()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        Ok(translation_object)
    }

    /// Upload a file to the server.
    ///
    /// # Arguments
    ///
    /// * `file` - The file to upload.
    ///
    /// # Returns
    ///
    /// A `Result` containing the file object or an error.
    pub async fn upload_file(&self, file: impl AsRef<Path>) -> Result<FileObject, LlamaEdgeError> {
        let abs_file_path = if file.as_ref().is_absolute() {
            file.as_ref().to_path_buf()
        } else {
            std::env::current_dir().unwrap().join(file.as_ref())
        };

        // check if the file exists
        if !abs_file_path.exists() {
            return Err(LlamaEdgeError::InvalidArgument(
                "The file does not exist".to_string(),
            ));
        }

        // get the filename
        let filename = abs_file_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        // get the file extension
        let file_extension = abs_file_path
            .extension()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let file = tokio::fs::read(abs_file_path)
            .await
            .map_err(|e| LlamaEdgeError::Operation(format!("Failed to read audio file: {}", e)))?;
        let file_part = multipart::Part::bytes(file)
            .file_name(filename)
            .mime_str(&format!("audio/{}", file_extension))
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        let form = multipart::Form::new().part("file", file_part);

        // upload the audio file
        let url = self.server_base_url.join("/v1/files")?;
        let response = reqwest::Client::new()
            .post(url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        // get the file object
        let file_object = response
            .json::<FileObject>()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        Ok(file_object)
    }

    /// List all available models.
    ///
    /// # Returns
    ///
    /// A `Result` containing the list of models or an error.
    pub async fn models(&self) -> Result<Vec<Model>, LlamaEdgeError> {
        let url = self.server_base_url.join("/v1/models")?;
        let response = reqwest::Client::new()
            .get(url)
            .send()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;
        let list_models_response = response
            .json::<ListModelsResponse>()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        Ok(list_models_response.data)
    }

    /// Compute embeddings for a given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to compute embeddings for.
    ///
    /// * `params` - The parameters for the embeddings.
    ///
    /// # Returns
    ///
    /// A `Result` containing the embeddings or an error.
    pub async fn embeddings(
        &self,
        input: InputText,
        params: EmbeddingsParams,
    ) -> Result<EmbeddingsResponse, LlamaEdgeError> {
        let url = self.server_base_url.join("/v1/embeddings")?;

        let request = EmbeddingRequest {
            input,
            model: params.model,
            encoding_format: Some(params.encoding_format),
            user: params.user,
            vdb_server_url: params.vdb_server_url,
            vdb_collection_name: params.vdb_collection_name,
            vdb_api_key: params.vdb_api_key,
        };

        let response = reqwest::Client::new()
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        let embeddings_response = response
            .json::<EmbeddingsResponse>()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        Ok(embeddings_response)
    }

    /// Create an image with the given prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt for the image.
    ///
    /// * `params` - The parameters for the image creation.
    ///
    /// # Returns
    ///
    /// A `Result` containing the list of images or an error.
    pub async fn create_image(
        &self,
        prompt: impl AsRef<str>,
        params: ImageCreateParams,
    ) -> Result<Vec<ImageObject>, LlamaEdgeError> {
        let url = self.server_base_url.join("/v1/images/generations")?;

        // build the request
        let mut builder = ImageCreateRequestBuilder::new(params.model, prompt.as_ref())
            .with_number_of_images(params.n)
            .with_response_format(params.response_format)
            .with_cfg_scale(params.cfg_scale)
            .with_sample_method(params.sample_method)
            .with_steps(params.steps)
            .with_image_size(params.height, params.width)
            .with_control_strength(params.control_strength)
            .with_seed(params.seed)
            .with_strength(params.strength)
            .with_scheduler(params.scheduler)
            .apply_canny_preprocessor(params.apply_canny_preprocessor)
            .with_style_ratio(params.style_ratio);
        if let Some(negative_prompt) = params.negative_prompt {
            builder = builder.with_negative_prompt(negative_prompt);
        }
        if let Some(user) = params.user {
            builder = builder.with_user(user);
        }
        if let Some(control_image) = params.control_image {
            builder = builder.with_control_image(control_image);
        }
        let request = builder.build();

        // send the request
        let response = reqwest::Client::new()
            .post(url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        let list_images_response = response
            .json::<ListImagesResponse>()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        Ok(list_images_response.data)
    }

    /// Edit the given image with the given prompt.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to edit.
    ///
    /// * `prompt` - The prompt for the image edit.
    ///
    /// * `params` - The parameters for the image edit.
    ///
    /// # Returns
    ///
    /// A `Result` containing the list of images or an error.
    pub async fn edit_image(
        &self,
        image: impl AsRef<Path>,
        prompt: impl AsRef<str>,
        params: ImageEditParams,
    ) -> Result<Vec<ImageObject>, LlamaEdgeError> {
        let abs_file_path = if image.as_ref().is_absolute() {
            image.as_ref().to_path_buf()
        } else {
            std::env::current_dir().unwrap().join(image.as_ref())
        };

        // check if the file exists
        if !abs_file_path.exists() {
            let error_message =
                format!("The image file does not exist: {}", abs_file_path.display());

            return Err(LlamaEdgeError::InvalidArgument(error_message));
        }

        // get the filename
        let filename = abs_file_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        // get the file extension
        let file_extension = abs_file_path
            .extension()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let file = tokio::fs::read(abs_file_path).await.map_err(|e| {
            LlamaEdgeError::Operation(format!("Failed to read the image file: {}", e))
        })?;

        let form = {
            let file_part = multipart::Part::bytes(file)
                .file_name(filename)
                .mime_str(&format!("image/{}", file_extension))
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let prompt_part = multipart::Part::text(prompt.as_ref().to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let model_part = multipart::Part::text(params.model.clone())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let n_part = multipart::Part::text(params.n.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let response_format_part = multipart::Part::text(params.response_format.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let cfg_scale_part = multipart::Part::text(params.cfg_scale.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let sample_method_part = multipart::Part::text(params.sample_method.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let steps_part = multipart::Part::text(params.steps.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let height_part = multipart::Part::text(params.height.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let width_part = multipart::Part::text(params.width.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let control_strength_part = multipart::Part::text(params.control_strength.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let seed_part = multipart::Part::text(params.seed.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let strength_part = multipart::Part::text(params.strength.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let scheduler_part = multipart::Part::text(params.scheduler.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let apply_canny_preprocessor_part =
                multipart::Part::text(params.apply_canny_preprocessor.to_string())
                    .mime_str("text/plain")
                    .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let style_ratio_part = multipart::Part::text(params.style_ratio.to_string())
                .mime_str("text/plain")
                .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

            let mut form = multipart::Form::new()
                .part("file", file_part)
                .part("prompt", prompt_part)
                .part("model", model_part)
                .part("n", n_part)
                .part("response_format", response_format_part)
                .part("cfg_scale", cfg_scale_part)
                .part("sample_method", sample_method_part)
                .part("steps", steps_part)
                .part("height", height_part)
                .part("width", width_part)
                .part("control_strength", control_strength_part)
                .part("seed", seed_part)
                .part("strength", strength_part)
                .part("scheduler", scheduler_part)
                .part("apply_canny_preprocessor", apply_canny_preprocessor_part)
                .part("style_ratio", style_ratio_part);

            if let Some(user) = params.user {
                let user_part = multipart::Part::text(user)
                    .mime_str("text/plain")
                    .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;
                form = form.part("user", user_part);
            }

            if let Some(negative_prompt) = params.negative_prompt {
                let negative_prompt_part = multipart::Part::text(negative_prompt)
                    .mime_str("text/plain")
                    .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;
                form = form.part("negative_prompt", negative_prompt_part);
            }

            if let Some(mask) = params.mask {
                let abs_mask_file_path = if mask.is_absolute() {
                    mask.to_path_buf()
                } else {
                    std::env::current_dir().unwrap().join(mask)
                };

                // check if the file exists
                if !abs_mask_file_path.exists() {
                    let error_message = format!(
                        "The mask image file does not exist: {}",
                        abs_mask_file_path.display()
                    );

                    return Err(LlamaEdgeError::InvalidArgument(error_message));
                }

                // get the filename
                let mask_filename = abs_mask_file_path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();

                // get the file extension
                let mask_file_extension = abs_mask_file_path
                    .extension()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();

                let mask_file = tokio::fs::read(abs_mask_file_path).await.map_err(|e| {
                    LlamaEdgeError::Operation(format!("Failed to read the image file: {}", e))
                })?;

                let mask_file_part = multipart::Part::bytes(mask_file)
                    .file_name(mask_filename)
                    .mime_str(&format!("image/{}", mask_file_extension))
                    .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

                form = form.part("mask", mask_file_part);
            }

            if let Some(control_image) = params.control_image {
                let abs_control_image_file_path = if control_image.is_absolute() {
                    control_image.to_path_buf()
                } else {
                    std::env::current_dir().unwrap().join(control_image)
                };

                // check if the file exists
                if !abs_control_image_file_path.exists() {
                    let error_message = format!(
                        "The control image file does not exist: {}",
                        abs_control_image_file_path.display()
                    );

                    return Err(LlamaEdgeError::InvalidArgument(error_message));
                }

                // get the filename
                let control_image_filename = abs_control_image_file_path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();

                // get the file extension
                let control_image_file_extension = abs_control_image_file_path
                    .extension()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();

                let control_image_file = tokio::fs::read(abs_control_image_file_path)
                    .await
                    .map_err(|e| {
                        LlamaEdgeError::Operation(format!("Failed to read the image file: {}", e))
                    })?;

                let control_image_file_part = multipart::Part::bytes(control_image_file)
                    .file_name(control_image_filename)
                    .mime_str(&format!("image/{}", control_image_file_extension))
                    .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

                form = form.part("control_image", control_image_file_part);
            }

            form
        };

        let url = self.server_base_url.join("/v1/images/edits")?;

        let response = reqwest::Client::new()
            .post(url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        let list_images_response = response
            .json::<ListImagesResponse>()
            .await
            .map_err(|e| LlamaEdgeError::Operation(e.to_string()))?;

        Ok(list_images_response.data)
    }
}
