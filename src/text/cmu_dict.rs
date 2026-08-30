use std::collections::HashMap;
use lazy_static::lazy_static;

const CMU_DATA: &str = include_str!("../../assets/cmudict-fast.rep");

lazy_static! {
    pub static ref CMU_DICT: HashMap<&'static str, Vec<&'static str>> = {
        let mut m = HashMap::with_capacity(135000);
        for line in CMU_DATA.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            if let Some(word) = parts.next() {
                // Remove variant like A(2)
                let base_word = if let Some(idx) = word.find('(') {
                    &word[..idx]
                } else {
                    word
                };
                let phonemes: Vec<&'static str> = parts.collect();
                m.entry(base_word).or_insert(phonemes);
            }
        }
        m
    };
}

/// Lookup English word in CMU dictionary or fallback to Arpabet approximation
pub fn get_english_phonemes(word: &str) -> Vec<String> {
    let upper = word.to_uppercase();
    if let Some(ph) = CMU_DICT.get(upper.as_str()) {
        return ph.iter().map(|&s| s.to_string()).collect();
    }

    // Simple letter-to-sound fallback
    let mut phones = Vec::new();
    for c in upper.chars() {
        match c {
            'A' => phones.push("EY1".to_string()),
            'B' => phones.push("B".to_string()),
            'C' => { phones.push("K".to_string()); },
            'D' => phones.push("D".to_string()),
            'E' => phones.push("IY1".to_string()),
            'F' => phones.push("F".to_string()),
            'G' => phones.push("G".to_string()),
            'H' => phones.push("HH".to_string()),
            'I' => phones.push("AY1".to_string()),
            'J' => phones.push("JH".to_string()),
            'K' => phones.push("K".to_string()),
            'L' => phones.push("L".to_string()),
            'M' => phones.push("M".to_string()),
            'N' => phones.push("N".to_string()),
            'O' => phones.push("OW1".to_string()),
            'P' => phones.push("P".to_string()),
            'Q' => { phones.push("K".to_string()); phones.push("W".to_string()); },
            'R' => phones.push("R".to_string()),
            'S' => phones.push("S".to_string()),
            'T' => phones.push("T".to_string()),
            'U' => phones.push("UW1".to_string()),
            'V' => phones.push("V".to_string()),
            'W' => phones.push("W".to_string()),
            'X' => { phones.push("K".to_string()); phones.push("S".to_string()); },
            'Y' => phones.push("Y".to_string()),
            'Z' => phones.push("Z".to_string()),
            _ => {}
        }
    }

    if phones.is_empty() {
        phones.push("UNK".to_string());
    }
    phones
}
