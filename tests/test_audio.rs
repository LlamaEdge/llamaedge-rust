use llamaedge::{params::TranscriptionParams, Client};

const SERVER_BASE_URL: &str = "http://localhost:8080";

#[tokio::test]
async fn test_transcribe() {
    let client = Client::new(SERVER_BASE_URL).unwrap();

    let result = client
        .transcribe("tests/assets/test.wav", &TranscriptionParams::default())
        .await;
    assert!(result.is_ok());

    let transcription = result.unwrap();
    let text = transcription.text.to_lowercase();
    assert!(text.contains("this is a test record for whisper.cpp"));
}
