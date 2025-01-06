use llamaedge::{params::TranslationParams, Client};

#[tokio::main]
async fn main() {
    const SERVER_BASE_URL: &str = "http://localhost:8080";

    let client = Client::new(SERVER_BASE_URL).unwrap();

    let translation_object = match client
        .translate(
            "tests/assets/test_zh.wav",
            "zh",
            TranslationParams::default(),
        )
        .await
    {
        Ok(to) => to,
        Err(e) => {
            println!("Error: {}", e);
            return;
        }
    };

    println!("{}", translation_object.text);
}
