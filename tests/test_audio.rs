use llamaedge::{
    params::{TranscriptionParams, TranslationParams},
    Client,
};

const SERVER_BASE_URL: &str = "http://localhost:12345";

#[tokio::test]
async fn test_audio_transcribe() {
    let client = Client::new(SERVER_BASE_URL).unwrap();

    let result = client
        .transcribe(
            "tests/assets/test.wav",
            "en",
            TranscriptionParams::default(),
        )
        .await;
    assert!(result.is_ok());

    let transcription = result.unwrap();
    let text = transcription.text.to_lowercase();
    assert!(text.contains("this is a test record for whisper.cpp"));
}

#[tokio::test]
async fn test_audio_translate() {
    let client = Client::new(SERVER_BASE_URL).unwrap();

    let result = client
        .translate(
            "tests/assets/test_zh.wav",
            "zh",
            TranslationParams::default(),
        )
        .await;
    assert!(result.is_ok());

    let translation = result.unwrap();
    let text = translation.text.to_lowercase();
    assert!(text.to_lowercase().contains("this is a chinese broadcast."));
}
