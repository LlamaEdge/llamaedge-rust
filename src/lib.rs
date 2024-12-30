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

use endpoints::chat::{
    ChatCompletionObject, ChatCompletionRequest, ChatCompletionRequestMessage, StreamOptions,
};
use error::LlamaEdgeError;
use futures::{stream::TryStream, StreamExt};
use params::ChatParams;
use url::Url;

/// Client for the LlamaEdge API.
pub struct Client {
    server_base_url: Url,
}
impl Client {
    /// Create a new client.
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
    pub fn server_base_url(&self) -> &Url {
        &self.server_base_url
    }

    /// Send a chat completion request.
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
    pub fn transcribe(&self, _audio: impl AsRef<str>) -> Result<String, LlamaEdgeError> {
        unimplemented!("Not implemented");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use endpoints::chat::{
        ChatCompletionChunk, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
        ChatCompletionUserMessage, ChatCompletionUserMessageContent,
    };

    const SERVER_BASE_URL: &str = "http://localhost:10086";

    #[tokio::test]
    async fn test_chat() {
        let client = Client::new(SERVER_BASE_URL).unwrap();

        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new(
                "You are a helpful assistant. Answer questions as concisely and accurately as possible.",
                None,
            ),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("What is the capital of France?".to_string()),
            None,
        ));
        messages.push(user_message);

        let result = client.chat(&messages[..], &ChatParams::default()).await;

        assert!(result.is_ok());
        let generation = result.unwrap();
        println!("{}", generation);
    }

    #[tokio::test]
    async fn test_chat_stream() {
        let client = Client::new(SERVER_BASE_URL).unwrap();

        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new(
                "You are a helpful assistant. Answer questions as concisely and accurately as possible.",
                None,
            ),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("What is the capital of France?".to_string()),
            None,
        ));
        messages.push(user_message);

        let result = client
            .chat_stream(&messages[..], &ChatParams::default())
            .await;
        assert!(result.is_ok());
        let mut stream = result.unwrap();

        // iterate over the stream
        let mut output = String::new();
        while let Some(item) = stream.next().await {
            if let Ok(event) = item {
                let event_parts = event
                    .split("data: ")
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<&str>>();

                for part in event_parts.iter() {
                    if *part == "[DONE]" {
                        break;
                    }

                    if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(part) {
                        if !chunk.choices.is_empty() {
                            if let Some(content) = &chunk.choices[0].delta.content {
                                let content = content.trim();
                                if !content.is_empty() {
                                    // append content to output
                                    output.push_str(content);
                                }
                            }
                        }
                    }
                }
            }
        }

        assert!(!output.is_empty());
        assert!(output.contains("Paris"));
        println!("output: {}", output);
    }
}
