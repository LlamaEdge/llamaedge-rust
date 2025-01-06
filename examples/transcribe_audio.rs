use llamaedge::{params::TranscriptionParams, Client};

#[tokio::main]
async fn main() {
    const SERVER_BASE_URL: &str = "http://localhost:8080";

    let client = Client::new(SERVER_BASE_URL).unwrap();

    let transcription_object = match client
        .transcribe(
            "tests/assets/test.wav",
            "en",
            TranscriptionParams::default(),
        )
        .await
    {
        Ok(to) => to,
        Err(e) => {
            println!("Error: {}", e);
            return;
        }
    };

    println!("{}", transcription_object.text);
}
