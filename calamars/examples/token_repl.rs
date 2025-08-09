use calamars::token::Token;
use std::io::{Write, stdin, stdout};

fn main() {
    println!("Enter line of Calamars to tokenize it!\n");
    loop {
        print!("\x1b[93m>>> \x1b[0m");
        let _ = stdout().flush();
        let mut buffer: String = String::new();
        stdin()
            .read_line(&mut buffer)
            .expect("Error reading user input");
        let tok = Token::tokenize_line(buffer);
        println!("{:?}", tok);
    }
}
