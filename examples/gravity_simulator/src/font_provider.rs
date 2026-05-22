use font_loader::system_fonts;
use nannou::text::Font;
pub struct FontProvider {
    bytes: Vec<u8>,
}

fn choose_font(fonts: Vec<String>) -> String {
    // Fonts with symbol ₴
    let preferred_fonts = ["DejaVu Sans Mono", "FreeMono", "FreeSans Mono"];

    for font in preferred_fonts {
        if fonts.contains(&font.into()) {
            return font.into();
        }
    }

    fonts.first().unwrap().into()
}

impl FontProvider {
    pub fn new() -> Self {
        let mut property = system_fonts::FontPropertyBuilder::new().monospace().build();
        let sys_fonts = system_fonts::query_specific(&mut property);
        let font_bytes = system_fonts::get(
            &system_fonts::FontPropertyBuilder::new()
                .family(&choose_font(sys_fonts))
                .build(),
        )
        .unwrap();

        Self {
            bytes: font_bytes.0,
        }
    }

    pub fn font(&'static self) -> Font {
        Font::from_bytes(&self.bytes).unwrap()
    }
}
