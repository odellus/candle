use std::env;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

fn main() {
    println!("cargo:rerun-if-changed=src/shaders");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("compiled_shaders.rs");

    let mut shader_code = String::from("// Auto-generated shader modules\n\n");
    shader_code.push_str("use std::collections::HashMap;\n\n");
    shader_code.push_str("pub struct CompiledShaders {\n");
    shader_code.push_str("    shaders: HashMap<&'static str, &'static [u8]>,\n");
    shader_code.push_str("}\n\n");
    shader_code.push_str("impl CompiledShaders {\n");
    shader_code.push_str("    pub fn new() -> Self {\n");
    shader_code.push_str("        let mut shaders = HashMap::new();\n");

    let shader_dir = Path::new("src/shaders");
    if shader_dir.exists() {
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );

        for entry in WalkDir::new(shader_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "comp"))
        {
            let path = entry.path();
            let file_name = path.file_stem().unwrap().to_str().unwrap();
            let shader_source = fs::read_to_string(path).unwrap();

            let binary_result = compiler
                .compile_into_spirv(
                    &shader_source,
                    shaderc::ShaderKind::Compute,
                    path.to_str().unwrap(),
                    "main",
                    Some(&options),
                )
                .unwrap();

            let spirv_bytes = binary_result.as_binary_u8();
            let const_name = format!("SHADER_{}", file_name.to_uppercase());

            shader_code.push_str(&format!(
                "        const {}: &[u8] = &{:?};\n",
                const_name, spirv_bytes
            ));
            shader_code.push_str(&format!(
                "        shaders.insert(\"{}\", {});\n",
                file_name, const_name
            ));
        }
    }

    shader_code.push_str("        Self { shaders }\n");
    shader_code.push_str("    }\n\n");
    shader_code.push_str("    pub fn get(&self, name: &str) -> Option<&'static [u8]> {\n");
    shader_code.push_str("        self.shaders.get(name).copied()\n");
    shader_code.push_str("    }\n");
    shader_code.push_str("}\n");

    fs::write(dest_path, shader_code).unwrap();
}
