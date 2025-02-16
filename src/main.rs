mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use crate::model::ChatSession;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        500,
        0.8,
        30,
        1.,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    // Load chat-optimized model 
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    
    // Initialize chat session
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let mut session = ChatSession::new(llama);

    // Interactive chat loop
    loop {
        // Read user input
        println!("\nUser input (type 'exit' to quit):");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        
        // Exit condition
        if input.eq_ignore_ascii_case("exit") {
            break;
        }

        // Add user message with English role
        session.add_user_message(input.to_string());
        
        // Build prompt with English template
        let prompt = session.build_prompt();
        let binding = tokenizer.encode(&*prompt, true).unwrap();
        let prompt_ids = binding.get_ids();

        // Generate response
        let output_ids = session.chat(
            prompt_ids,
            100,    // max_len
            0.8,    // top_p
            30,     // top_k
            1.0,    // temperature
        );
        
        // Decode and display response
        let response = tokenizer.decode(&output_ids, true).unwrap();
        println!("\nAssistant:\n{}", response.trim());
        
        // Add assistant response with English role
        session.add_assistant_message(response);
    }
}
