use llamaedge::{params::ImageCreateParams, Client};

#[tokio::main]
async fn main() {
    const SERVER_BASE_URL: &str = "http://localhost:8080";

    let client = Client::new(SERVER_BASE_URL).unwrap();

    let image_object_vec = match client
        .create_image("A lovely dog", ImageCreateParams::default())
        .await
    {
        Ok(image_object_vec) => image_object_vec,
        Err(e) => {
            println!("Error: {}", e);
            return;
        }
    };

    println!("{:?}", image_object_vec[0]);
}
