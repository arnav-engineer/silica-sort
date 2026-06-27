use raw_cpuid::CpuId;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    AVX512,
    AVX2,
    #[cfg(target_arch = "aarch64")]
    NEON,
    Scalar,
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub simd_level: SimdLevel,
}

impl SystemInfo {
    pub fn detect() -> Self {
        static SYSTEM_INFO: OnceLock<SystemInfo> = OnceLock::new();
        SYSTEM_INFO.get_or_init(Self::detect_uncached).clone()
    }

    fn detect_uncached() -> Self {
        let cpuid = CpuId::new();

        // Detect L1 Cache Size (Data Cache)
        // This is a heuristic based on common CPUID leaves.
        let l1 = cpuid
            .get_cache_parameters()
            .and_then(|c| {
                c.filter(|p| {
                    p.level() == 1
                        && (p.cache_type() == raw_cpuid::CacheType::Data
                            || p.cache_type() == raw_cpuid::CacheType::Unified)
                })
                .map(|p| p.sets() * p.associativity() * p.coherency_line_size())
                .max()
            })
            .unwrap_or(32 * 1024); // Fallback

        // Detect L2 Cache Size
        let l2 = cpuid
            .get_cache_parameters()
            .and_then(|c| {
                c.filter(|p| p.level() == 2)
                    .map(|p| p.sets() * p.associativity() * p.coherency_line_size())
                    .max()
            })
            .unwrap_or(256 * 1024); // Fallback

        let simd_level = Self::detect_simd();

        Self {
            l1_cache_size: l1,
            l2_cache_size: l2,
            simd_level,
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect_simd() -> SimdLevel {
        if is_x86_feature_detected!("avx512f") {
            SimdLevel::AVX512
        } else if is_x86_feature_detected!("avx2") {
            SimdLevel::AVX2
        } else {
            SimdLevel::Scalar
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_simd() -> SimdLevel {
        // NEON is mandatory on AArch64 relative to standard baseline,
        // though strictly speaking we could check Features.
        SimdLevel::NEON
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect_simd() -> SimdLevel {
        SimdLevel::Scalar
    }
}
