use crate::flds::field::{Field, FieldDim};
use crate::{Float, Sim};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

pub struct Fft2D {
    field_size: FieldDim,
    fft_x: std::sync::Arc<dyn rustfft::Fft<Float>>,
    ifft_x: std::sync::Arc<dyn rustfft::Fft<Float>>,
    fft_y: std::sync::Arc<dyn rustfft::Fft<Float>>,
    ifft_y: std::sync::Arc<dyn rustfft::Fft<Float>>,
    xscratch: Vec<Complex<Float>>,
    yscratch: Vec<Complex<Float>>,
    wrkspace: Vec<Complex<Float>>,
}

impl Fft2D {
    pub fn new(sim: &Sim) -> Fft2D {
        let mut planner = FftPlanner::new();
        let field_size = FieldDim {
            size_x: sim.size_x,
            size_y: sim.size_y,
        };
        let fft_x = planner.plan_fft_forward(field_size.size_x);
        let ifft_x = planner.plan_fft_inverse(field_size.size_x);
        let fft_y = planner.plan_fft_forward(field_size.size_y);
        let ifft_y = planner.plan_fft_inverse(field_size.size_y);
        let xscratch = vec![Complex::zero(); fft_x.get_outofplace_scratch_len()];
        let yscratch = vec![Complex::zero(); fft_y.get_outofplace_scratch_len()];
        let wrkspace = vec![Complex::zero(); sim.size_x * sim.size_y];

        Fft2D {
            field_size,
            fft_x,
            ifft_x,
            fft_y,
            ifft_y,
            xscratch,
            yscratch,
            wrkspace,
        }
    }
    fn transpose_out_of_place(
        in_vec: &mut Vec<Complex<Float>>,
        out_vec: &mut Vec<Complex<Float>>,
        dim: &mut FieldDim,
    ) {
        // check to make sure the two vecs are the same size
        let size_x = dim.size_x;
        let size_y = dim.size_y;

        if !cfg!(feature = "unchecked") {
            assert_eq!(in_vec.len(), out_vec.len());
            assert_eq!(in_vec.len(), size_x * size_y);
        }
        for i in 0..size_y {
            for j in 0..size_x {
                unsafe {
                    // If you don't trust this unsafe section,
                    // run the code with the checked feature
                    // len(out_fld) == len(in_fld)
                    // && size_y * size_x == len(out_fld)
                    *out_vec.get_unchecked_mut(j * size_y + i) =
                        *in_vec.get_unchecked(i * size_x + j);
                }
                // bounds checked version
                // out_fld[i * sim.size_x + j] = in_fld[j * sim.size_y + i];
            }
        }
        dim.size_x = size_y;
        dim.size_y = size_x;
    }

    pub fn fft(&mut self, fld: &mut Field) {
        if !cfg!(feature = "unchecked") {
            assert_eq!(self.field_size, fld.no_ghost_dim);
        }

        self.fft_x.process_outofplace_with_scratch(
            &mut fld.spectral,
            &mut self.wrkspace,
            &mut self.xscratch,
        );

        Fft2D::transpose_out_of_place(&mut self.wrkspace, &mut fld.spectral, &mut self.field_size);
        self.fft_y.process_outofplace_with_scratch(
            &mut fld.spectral,
            &mut self.wrkspace,
            &mut self.yscratch,
        );

        Fft2D::transpose_out_of_place(&mut self.wrkspace, &mut fld.spectral, &mut self.field_size);
    }

    pub fn inv_fft(&mut self, fld: &mut Field) {
        if !cfg!(feature = "unchecked") {
            assert_eq!(self.field_size, fld.no_ghost_dim);
        }

        self.ifft_x.process_outofplace_with_scratch(
            &mut fld.spectral,
            &mut self.wrkspace,
            &mut self.xscratch,
        );

        Fft2D::transpose_out_of_place(&mut self.wrkspace, &mut fld.spectral, &mut self.field_size);
        self.ifft_y.process_outofplace_with_scratch(
            &mut fld.spectral,
            &mut self.wrkspace,
            &mut self.yscratch,
        );

        Fft2D::transpose_out_of_place(&mut self.wrkspace, &mut fld.spectral, &mut self.field_size);

        let norm = (fld.spectral.len() as Float).powi(-1);
        for v in fld.spectral.iter_mut() {
            *v *= norm;
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{build_test_sim, flds::field::Pos, E_TOL};

    fn get_2d_input() -> Vec<Complex<Float>> {
        let input: Vec<Complex<Float>> = vec![
            Complex::new(0.7739560485559633, 0.0),
            Complex::new(0.4388784397520523, 0.0),
            Complex::new(0.8585979199113825, 0.0),
            Complex::new(0.6973680290593639, 0.0),
            Complex::new(0.09417734788764953, 0.0),
            Complex::new(0.9756223516367559, 0.0),
            Complex::new(0.761139701990353, 0.0),
            Complex::new(0.7860643052769538, 0.0),
            Complex::new(0.12811363267554587, 0.0),
            Complex::new(0.45038593789556713, 0.0),
            Complex::new(0.37079802423258124, 0.0),
            Complex::new(0.9267649888486018, 0.0),
            Complex::new(0.6438651200806645, 0.0),
            Complex::new(0.82276161327083, 0.0),
            Complex::new(0.44341419882733113, 0.0),
            Complex::new(0.2272387217847769, 0.0),
            Complex::new(0.5545847870158348, 0.0),
            Complex::new(0.06381725610417532, 0.0),
            Complex::new(0.8276311719925821, 0.0),
            Complex::new(0.6316643991220648, 0.0),
            Complex::new(0.7580877400853738, 0.0),
            Complex::new(0.35452596812986836, 0.0),
            Complex::new(0.9706980243949033, 0.0),
            Complex::new(0.8931211213221977, 0.0),
            Complex::new(0.7783834970737619, 0.0),
            Complex::new(0.19463870785196757, 0.0),
            Complex::new(0.4667210037270342, 0.0),
            Complex::new(0.04380376578722878, 0.0),
            Complex::new(0.15428949206754783, 0.0),
            Complex::new(0.6830489532424546, 0.0),
            Complex::new(0.7447621559078171, 0.0),
            Complex::new(0.96750973243421, 0.0),
            Complex::new(0.32582535813815194, 0.0),
            Complex::new(0.3704597060348689, 0.0),
            Complex::new(0.4695558112758079, 0.0),
            Complex::new(0.1894713590842857, 0.0),
            Complex::new(0.12992150533547164, 0.0),
            Complex::new(0.47570492622593374, 0.0),
            Complex::new(0.2269093490508841, 0.0),
            Complex::new(0.6698139946825103, 0.0),
            Complex::new(0.43715191887233074, 0.0),
            Complex::new(0.8326781960578374, 0.0),
            Complex::new(0.7002651020022491, 0.0),
            Complex::new(0.31236664138204107, 0.0),
            Complex::new(0.8322598013952011, 0.0),
            Complex::new(0.8047643574968019, 0.0),
            Complex::new(0.38747837903017446, 0.0),
            Complex::new(0.2883281039302441, 0.0),
            Complex::new(0.6824955039749755, 0.0),
            Complex::new(0.1397524836093098, 0.0),
            Complex::new(0.19990820247510832, 0.0),
            Complex::new(0.007362269751005512, 0.0),
            Complex::new(0.7869243775021384, 0.0),
            Complex::new(0.6648508565920321, 0.0),
            Complex::new(0.7051653786263351, 0.0),
            Complex::new(0.7807290310219679, 0.0),
            Complex::new(0.45891577553833995, 0.0),
            Complex::new(0.5687411959528937, 0.0),
            Complex::new(0.13979699812765745, 0.0),
            Complex::new(0.11453007353597344, 0.0),
            Complex::new(0.6684029617904717, 0.0),
            Complex::new(0.4710962061431325, 0.0),
            Complex::new(0.5652361064811888, 0.0),
            Complex::new(0.7649988574160256, 0.0),
            Complex::new(0.6347183200005908, 0.0),
            Complex::new(0.5535794006579958, 0.0),
            Complex::new(0.5592071607454135, 0.0),
            Complex::new(0.3039500980626122, 0.0),
            Complex::new(0.030817834567939406, 0.0),
            Complex::new(0.43671738923236236, 0.0),
            Complex::new(0.2145846728195292, 0.0),
            Complex::new(0.40852864372463615, 0.0),
            Complex::new(0.8534030732681661, 0.0),
            Complex::new(0.23393948586534075, 0.0),
            Complex::new(0.05830274168906602, 0.0),
            Complex::new(0.28138389202199654, 0.0),
            Complex::new(0.2935937577666836, 0.0),
            Complex::new(0.6619165147268951, 0.0),
            Complex::new(0.5570321523412783, 0.0),
            Complex::new(0.7838982091064135, 0.0),
            Complex::new(0.6643135403273875, 0.0),
            Complex::new(0.4063868614400705, 0.0),
            Complex::new(0.8140203846660347, 0.0),
            Complex::new(0.1669729199077039, 0.0),
            Complex::new(0.022712073133860478, 0.0),
            Complex::new(0.09004786077564175, 0.0),
            Complex::new(0.7223593505964503, 0.0),
            Complex::new(0.4618772302513874, 0.0),
            Complex::new(0.1612717790336018, 0.0),
            Complex::new(0.5010447751033635, 0.0),
            Complex::new(0.15231210271316842, 0.0),
            Complex::new(0.696320375077736, 0.0),
            Complex::new(0.4461562755740307, 0.0),
            Complex::new(0.3810212260964825, 0.0),
            Complex::new(0.3015120891478765, 0.0),
            Complex::new(0.6302825931188885, 0.0),
            Complex::new(0.3618126105533904, 0.0),
            Complex::new(0.087649919316101, 0.0),
            Complex::new(0.11800590212051532, 0.0),
            Complex::new(0.9618976645495145, 0.0),
            Complex::new(0.908580690707607, 0.0),
            Complex::new(0.6997071338107496, 0.0),
            Complex::new(0.2658699614595196, 0.0),
            Complex::new(0.9691763773477239, 0.0),
            Complex::new(0.7787509039657946, 0.0),
            Complex::new(0.7168901891589956, 0.0),
            Complex::new(0.44936150214378867, 0.0),
            Complex::new(0.272241561845159, 0.0),
            Complex::new(0.0963909621534993, 0.0),
            Complex::new(0.9026023965438417, 0.0),
            Complex::new(0.45577628983361107, 0.0),
            Complex::new(0.20236336479523032, 0.0),
            Complex::new(0.3059566241506525, 0.0),
            Complex::new(0.579219568941896, 0.0),
            Complex::new(0.1767727829392317, 0.0),
            Complex::new(0.8566142840923755, 0.0),
            Complex::new(0.7585195298352101, 0.0),
            Complex::new(0.7194629559509368, 0.0),
            Complex::new(0.4320930397751037, 0.0),
            Complex::new(0.6273088407024432, 0.0),
            Complex::new(0.5840979689127356, 0.0),
            Complex::new(0.64984660155482, 0.0),
            Complex::new(0.08444432113988909, 0.0),
            Complex::new(0.41580740217060963, 0.0),
            Complex::new(0.04161417386189248, 0.0),
            Complex::new(0.49399081924451893, 0.0),
            Complex::new(0.3298612123327853, 0.0),
            Complex::new(0.1445241888660469, 0.0),
            Complex::new(0.10340296772255164, 0.0),
            Complex::new(0.587644572177712, 0.0),
            Complex::new(0.1705929685368861, 0.0),
            Complex::new(0.9251201183767972, 0.0),
            Complex::new(0.581061139700395, 0.0),
            Complex::new(0.34686980453483707, 0.0),
            Complex::new(0.5909154914814168, 0.0),
            Complex::new(0.022803871029697498, 0.0),
            Complex::new(0.9585592132414453, 0.0),
            Complex::new(0.48230343694290023, 0.0),
            Complex::new(0.7827352272502862, 0.0),
            Complex::new(0.08272999992243857, 0.0),
            Complex::new(0.48665833083816035, 0.0),
            Complex::new(0.4907069943545209, 0.0),
            Complex::new(0.9378264549749828, 0.0),
            Complex::new(0.5717280523760754, 0.0),
            Complex::new(0.4734894010569538, 0.0),
            Complex::new(0.2669756630918936, 0.0),
            Complex::new(0.3315689973425522, 0.0),
            Complex::new(0.5206724024715378, 0.0),
            Complex::new(0.4389114603050467, 0.0),
            Complex::new(0.021612079880330426, 0.0),
            Complex::new(0.8262919241943578, 0.0),
            Complex::new(0.8961607718397667, 0.0),
            Complex::new(0.14024908899861077, 0.0),
            Complex::new(0.5540361435390494, 0.0),
            Complex::new(0.10857574113544355, 0.0),
            Complex::new(0.6722400930398117, 0.0),
            Complex::new(0.2812337838390083, 0.0),
            Complex::new(0.6594226346919018, 0.0),
            Complex::new(0.7269946142868826, 0.0),
            Complex::new(0.768647491917657, 0.0),
            Complex::new(0.10774094595589656, 0.0),
            Complex::new(0.9160118451376079, 0.0),
            Complex::new(0.23021399089488082, 0.0),
            Complex::new(0.03741255617617978, 0.0),
            Complex::new(0.5548524693914834, 0.0),
            Complex::new(0.37092228386243875, 0.0),
            Complex::new(0.8297897431324132, 0.0),
            Complex::new(0.8082514720643018, 0.0),
            Complex::new(0.31713889282271535, 0.0),
            Complex::new(0.952899395069745, 0.0),
            Complex::new(0.2909178381401186, 0.0),
            Complex::new(0.5150571292317145, 0.0),
            Complex::new(0.25596509056760275, 0.0),
            Complex::new(0.9360435700489633, 0.0),
            Complex::new(0.16460781758201815, 0.0),
            Complex::new(0.04491061939232899, 0.0),
            Complex::new(0.43509706000303794, 0.0),
            Complex::new(0.992375564055837, 0.0),
            Complex::new(0.891677266254914, 0.0),
            Complex::new(0.7486080194569492, 0.0),
            Complex::new(0.8907924908785249, 0.0),
            Complex::new(0.8934466396978632, 0.0),
            Complex::new(0.5188583603864491, 0.0),
            Complex::new(0.315929051830793, 0.0),
            Complex::new(0.772012432110988, 0.0),
            Complex::new(0.6616612631677611, 0.0),
            Complex::new(0.37365772887371007, 0.0),
            Complex::new(0.09446666806151527, 0.0),
            Complex::new(0.746789611349026, 0.0),
            Complex::new(0.2624605159228647, 0.0),
            Complex::new(0.9368131505337792, 0.0),
            Complex::new(0.24097057500568475, 0.0),
            Complex::new(0.12275793241148603, 0.0),
            Complex::new(0.8311126721249061, 0.0),
            Complex::new(0.15328431662449404, 0.0),
            Complex::new(0.1792683081577391, 0.0),
            Complex::new(0.5993827915208435, 0.0),
            Complex::new(0.8745620408374645, 0.0),
            Complex::new(0.19643466571457324, 0.0),
            Complex::new(0.31032367290009477, 0.0),
            Complex::new(0.7774048382411776, 0.0),
            Complex::new(0.9718264260609674, 0.0),
            Complex::new(0.5007411862023423, 0.0),
            Complex::new(0.14389750255125078, 0.0),
            Complex::new(0.013936287708201545, 0.0),
            Complex::new(0.22965602999885526, 0.0),
            Complex::new(0.13182221778652103, 0.0),
            Complex::new(0.6776586736128575, 0.0),
            Complex::new(0.12183250462853112, 0.0),
            Complex::new(0.506329931620633, 0.0),
            Complex::new(0.6942624356428865, 0.0),
            Complex::new(0.5811166092209024, 0.0),
            Complex::new(0.19977565166005762, 0.0),
            Complex::new(0.8041245261822627, 0.0),
            Complex::new(0.7154071296158017, 0.0),
            Complex::new(0.7389840039155418, 0.0),
            Complex::new(0.13105775155731325, 0.0),
            Complex::new(0.12375380365034461, 0.0),
            Complex::new(0.9275625510065076, 0.0),
            Complex::new(0.39757819382494064, 0.0),
            Complex::new(0.30094869178093975, 0.0),
            Complex::new(0.4885840453515333, 0.0),
            Complex::new(0.6628642127635824, 0.0),
            Complex::new(0.9556232570469699, 0.0),
            Complex::new(0.286446226882055, 0.0),
            Complex::new(0.9248084293120271, 0.0),
            Complex::new(0.024859491386256316, 0.0),
            Complex::new(0.5551980423268247, 0.0),
            Complex::new(0.6339751116810851, 0.0),
            Complex::new(0.1058974037507533, 0.0),
            Complex::new(0.14033959706391264, 0.0),
            Complex::new(0.41911431931630383, 0.0),
            Complex::new(0.9662319121431817, 0.0),
            Complex::new(0.5960425532343728, 0.0),
            Complex::new(0.9330232216002112, 0.0),
            Complex::new(0.8043609156129707, 0.0),
            Complex::new(0.4673816015552915, 0.0),
            Complex::new(0.7847634492521874, 0.0),
            Complex::new(0.017836783976987736, 0.0),
            Complex::new(0.10914399676573494, 0.0),
            Complex::new(0.8294286148827363, 0.0),
            Complex::new(0.7968170883251617, 0.0),
            Complex::new(0.2326407419664337, 0.0),
            Complex::new(0.5307695905990535, 0.0),
            Complex::new(0.6060158207000109, 0.0),
            Complex::new(0.8677389537760112, 0.0),
            Complex::new(0.6031071573387257, 0.0),
            Complex::new(0.4125715692736803, 0.0),
            Complex::new(0.3741840434071827, 0.0),
            Complex::new(0.4258820863735099, 0.0),
            Complex::new(0.6519310255799742, 0.0),
            Complex::new(0.8674906317523249, 0.0),
            Complex::new(0.45389688207629975, 0.0),
            Complex::new(0.24783956295135812, 0.0),
            Complex::new(0.23666236299758114, 0.0),
            Complex::new(0.7460142802434464, 0.0),
            Complex::new(0.8165687634239104, 0.0),
            Complex::new(0.10527807985412496, 0.0),
            Complex::new(0.06655885695517816, 0.0),
            Complex::new(0.5944336637564518, 0.0),
            Complex::new(0.14617324419269828, 0.0),
            Complex::new(0.8246641904563413, 0.0),
            Complex::new(0.31033467392407443, 0.0),
            Complex::new(0.14387193297114265, 0.0),
            Complex::new(0.9209704724874502, 0.0),
            Complex::new(0.16553172273527816, 0.0),
            Complex::new(0.28472008233793555, 0.0),
            Complex::new(0.15361339519205863, 0.0),
            Complex::new(0.11549006366535497, 0.0),
            Complex::new(0.021148016336440034, 0.0),
            Complex::new(0.05539540916426011, 0.0),
            Complex::new(0.1746414709358527, 0.0),
            Complex::new(0.05338193262717539, 0.0),
            Complex::new(0.5911438161109712, 0.0),
            Complex::new(0.6807145267995064, 0.0),
            Complex::new(0.39363045683202824, 0.0),
            Complex::new(0.3179910969520494, 0.0),
            Complex::new(0.5045262370222301, 0.0),
            Complex::new(0.8750049422346086, 0.0),
            Complex::new(0.851131626822206, 0.0),
            Complex::new(0.04347506201279194, 0.0),
            Complex::new(0.18149840959652408, 0.0),
            Complex::new(0.23674487110439602, 0.0),
            Complex::new(0.24938757583221183, 0.0),
            Complex::new(0.5712326517427727, 0.0),
            Complex::new(0.4162624257031923, 0.0),
            Complex::new(0.04925411992759399, 0.0),
            Complex::new(0.3736141384595716, 0.0),
        ];
        let sim = build_test_sim();
        assert_eq!(sim.size_x, 24);
        assert_eq!(sim.size_y, 12);
        assert_eq!(input.len(), 12 * 24);

        input
    }
    #[test]
    fn forward_fft() {
        // Test that the RustFFT agrees with Numpy
        let mut input: Vec<Complex<Float>> = vec![
            Complex::new(0.7739560485559633, 0.0),
            Complex::new(0.4388784397520523, 0.0),
            Complex::new(0.8585979199113825, 0.0),
            Complex::new(0.6973680290593639, 0.0),
            Complex::new(0.09417734788764953, 0.0),
            Complex::new(0.9756223516367559, 0.0),
            Complex::new(0.761139701990353, 0.0),
            Complex::new(0.7860643052769538, 0.0),
            Complex::new(0.12811363267554587, 0.0),
            Complex::new(0.45038593789556713, 0.0),
            Complex::new(0.37079802423258124, 0.0),
            Complex::new(0.9267649888486018, 0.0),
            Complex::new(0.6438651200806645, 0.0),
            Complex::new(0.82276161327083, 0.0),
            Complex::new(0.44341419882733113, 0.0),
            Complex::new(0.2272387217847769, 0.0),
            Complex::new(0.5545847870158348, 0.0),
            Complex::new(0.06381725610417532, 0.0),
            Complex::new(0.8276311719925821, 0.0),
            Complex::new(0.6316643991220648, 0.0),
            Complex::new(0.7580877400853738, 0.0),
            Complex::new(0.35452596812986836, 0.0),
            Complex::new(0.9706980243949033, 0.0),
            Complex::new(0.8931211213221977, 0.0),
        ];

        let out: Vec<Complex<Float>> = vec![
            Complex::new(14.453276849853372, 0.0),
            Complex::new(1.151341674477084, -0.23629347172720028),
            Complex::new(0.9240323845693825, 0.588395230248971),
            Complex::new(-1.1349266739157249, 0.7195873283354851),
            Complex::new(1.9571847859015015, -0.0155870261428106),
            Complex::new(-0.3053291301765327, -0.3228826230849507),
            Complex::new(-1.2794943650481012, 1.0562299986247101),
            Complex::new(-1.0230989237260295, 1.2480143960716876),
            Complex::new(-0.12247237569297761, 0.3272816248868606),
            Complex::new(1.0559751943030367, 0.48313771018340235),
            Complex::new(-0.15738713556020276, 0.1087502339196228),
            Complex::new(1.0365834298899586, -2.442588929049296),
            Complex::new(-0.08314941455304314, 0.0),
            Complex::new(1.0365834298899586, 2.442588929049296),
            Complex::new(-0.15738713556020276, -0.1087502339196228),
            Complex::new(1.0559751943030367, -0.48313771018340235),
            Complex::new(-0.12247237569297761, -0.3272816248868606),
            Complex::new(-1.0230989237260295, -1.2480143960716876),
            Complex::new(-1.2794943650481012, -1.0562299986247101),
            Complex::new(-0.3053291301765327, 0.3228826230849507),
            Complex::new(1.9571847859015015, 0.0155870261428106),
            Complex::new(-1.1349266739157249, -0.7195873283354851),
            Complex::new(0.9240323845693825, -0.588395230248971),
            Complex::new(1.151341674477084, 0.23629347172720028),
        ];
        let mut wrkspace: Vec<Complex<Float>> = vec![Complex::zero(); input.len()];
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(input.len());
        let mut xscratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];

        assert_eq!(input.len(), wrkspace.len());

        fft.process_outofplace_with_scratch(&mut input, &mut wrkspace, &mut xscratch);
        assert_eq!(out.len(), wrkspace.len());
        for (v1, v2) in wrkspace.iter().zip(out) {
            assert!((v1.re - v2.re).abs() < E_TOL);
            assert!((v1.im - v2.im).abs() < E_TOL);
        }
    }

    #[test]
    fn fft_1d_roundtrip() {
        let mut input: Vec<Complex<Float>> = vec![
            Complex::new(14.453276849853372, 0.0),
            Complex::new(1.151341674477084, -0.23629347172720028),
            Complex::new(0.9240323845693825, 0.588395230248971),
            Complex::new(-1.1349266739157249, 0.7195873283354851),
            Complex::new(1.9571847859015015, -0.0155870261428106),
            Complex::new(-0.3053291301765327, -0.3228826230849507),
            Complex::new(-1.2794943650481012, 1.0562299986247101),
            Complex::new(-1.0230989237260295, 1.2480143960716876),
            Complex::new(-0.12247237569297761, 0.3272816248868606),
            Complex::new(1.0559751943030367, 0.48313771018340235),
            Complex::new(-0.15738713556020276, 0.1087502339196228),
            Complex::new(1.0365834298899586, -2.442588929049296),
            Complex::new(-0.08314941455304314, 0.0),
            Complex::new(1.0365834298899586, 2.442588929049296),
            Complex::new(-0.15738713556020276, -0.1087502339196228),
            Complex::new(1.0559751943030367, -0.48313771018340235),
            Complex::new(-0.12247237569297761, -0.3272816248868606),
            Complex::new(-1.0230989237260295, -1.2480143960716876),
            Complex::new(-1.2794943650481012, -1.0562299986247101),
            Complex::new(-0.3053291301765327, 0.3228826230849507),
            Complex::new(1.9571847859015015, 0.0155870261428106),
            Complex::new(-1.1349266739157249, -0.7195873283354851),
            Complex::new(0.9240323845693825, -0.588395230248971),
            Complex::new(1.151341674477084, 0.23629347172720028),
        ];
        let expected_out = input.clone();
        let mut wrkspace: Vec<Complex<Float>> = vec![Complex::zero(); input.len()];
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(input.len());
        let ifft = planner.plan_fft_inverse(input.len());
        let mut scratch_1 = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
        let mut scratch_2 = vec![Complex::zero(); ifft.get_outofplace_scratch_len()];

        assert_eq!(input.len(), wrkspace.len());

        fft.process_outofplace_with_scratch(&mut input, &mut wrkspace, &mut scratch_1);
        // zero out input
        //
        for v in input.iter_mut() {
            *v *= 0.0;
        }
        // fft is un-normed so we need to norm it.
        let norm = input.len() as Float;
        ifft.process_outofplace_with_scratch(&mut wrkspace, &mut input, &mut scratch_2);
        for (v1, v2) in input.iter().zip(expected_out) {
            assert!((v1.re - v2.re * norm).abs() < E_TOL);
            assert!((v1.im - v2.im * norm).abs() < E_TOL);
        }
    }
    #[test]
    fn forward_fft2d() {
        // We compare our fft2 to numpy's implementation.
        let input = get_2d_input();
        let out: Vec<Complex<Float>> = vec![
            Complex::new(138.5114607251621, 0.0),
            Complex::new(-1.27440962315266, 0.753048308479376),
            Complex::new(-2.976660161059959, 5.064509803811594),
            Complex::new(3.246395766149941, 5.87134631796019),
            Complex::new(1.2575392304413482, 3.258384729843708),
            Complex::new(3.714827703116214, -1.8362091552886115),
            Complex::new(-0.05917777470758012, -3.097504801014835),
            Complex::new(2.042983399473313, 0.7584328570319661),
            Complex::new(3.0626271208529943, 0.3372921309564949),
            Complex::new(0.5664111994986987, 0.6758089296417115),
            Complex::new(2.9076022780396773, 1.9743964478727989),
            Complex::new(4.2726656682831905, -6.2022485119512085),
            Complex::new(-8.137267917648913, 0.0),
            Complex::new(4.2726656682831905, 6.2022485119512085),
            Complex::new(2.9076022780396773, -1.9743964478727989),
            Complex::new(0.5664111994986987, -0.6758089296417115),
            Complex::new(3.0626271208529943, -0.3372921309564949),
            Complex::new(2.042983399473313, -0.7584328570319661),
            Complex::new(-0.05917777470758012, 3.097504801014835),
            Complex::new(3.714827703116214, 1.8362091552886115),
            Complex::new(1.2575392304413482, -3.258384729843708),
            Complex::new(3.246395766149941, -5.87134631796019),
            Complex::new(-2.976660161059959, -5.064509803811594),
            Complex::new(-1.27440962315266, -0.753048308479376),
            Complex::new(-1.3717992651530322, 0.4889251692928045),
            Complex::new(0.22704221699843447, -1.1641261321031933),
            Complex::new(-0.631000565802627, 1.2481053818696088),
            Complex::new(5.7353056258742745, -0.3140966750394031),
            Complex::new(2.321351756601554, -5.779177535488895),
            Complex::new(7.821013356793824, -4.828722008992704),
            Complex::new(1.8517301932817827, 2.392592405787043),
            Complex::new(2.908989148177296, -5.1175738846889445),
            Complex::new(4.115157876324687, 2.2409594223568083),
            Complex::new(2.467673303314676, 5.845782635321265),
            Complex::new(1.9130057483669187, 0.13512884581887286),
            Complex::new(2.3541456077858585, 1.4331453706786732),
            Complex::new(3.1443637804160414, -2.1820688693695383),
            Complex::new(0.8776658356175657, -2.065150780332987),
            Complex::new(4.237273462279213, -4.143807763539963),
            Complex::new(2.824412135830933, -5.962685818348512),
            Complex::new(2.3682628894254423, 2.23008329676306),
            Complex::new(-1.6584579884976751, -6.310483857378658),
            Complex::new(-2.0834510892728146, -4.154462997012521),
            Complex::new(-0.7055687959474743, 3.593264353832105),
            Complex::new(3.789956544025782, 0.8089187417597645),
            Complex::new(-5.6096175895169775, -2.307740061532588),
            Complex::new(-1.8085495290827898, 3.5789274626261838),
            Complex::new(1.0390903665791136, -0.4092004505236395),
            Complex::new(2.573481318982692, -2.542591689870939),
            Complex::new(-0.2608678824289141, 4.038843854571754),
            Complex::new(9.200340526131951, 3.674620173296757),
            Complex::new(-0.8454581277394115, -0.7596011837107017),
            Complex::new(3.4396356375736, -0.4342370304384102),
            Complex::new(-1.8693468194234835, 0.46998827429938417),
            Complex::new(-8.426737895239928, 1.477059762741725),
            Complex::new(0.2548853901484276, 6.252250833184429),
            Complex::new(-0.4272378426861331, 2.472616834988732),
            Complex::new(0.9431463653346317, 1.4900894364351855),
            Complex::new(-4.651734564401249, -3.957622481137731),
            Complex::new(5.195168946325346, -5.599599699993183),
            Complex::new(2.612825067014143, -0.6262449302657651),
            Complex::new(3.26214566690222, 4.4555909540179455),
            Complex::new(-0.8401788927293691, 3.409461120256584),
            Complex::new(-4.499470544262742, 5.785691094584783),
            Complex::new(-0.8057950339464541, -2.028750060451022),
            Complex::new(2.6252451595071933, 0.6551883981264515),
            Complex::new(0.7673491433916113, 1.8345734779370055),
            Complex::new(-3.53788359364677, 1.9680763523721678),
            Complex::new(2.071398814561847, -2.4500203908496303),
            Complex::new(-2.343176995718726, -4.8101493001419975),
            Complex::new(8.61082169972553, 4.801900847453371),
            Complex::new(0.51222948521139, -3.0422742756607963),
            Complex::new(3.9380430471136094, -2.2336101756446247),
            Complex::new(6.080940845573007, -2.324482277023904),
            Complex::new(-1.7813777739001353, 11.556784291451603),
            Complex::new(-3.8248062027390475, -2.5250116691631277),
            Complex::new(-0.7553982110067918, -4.521577531461968),
            Complex::new(-1.79799016628888, 11.814947622610552),
            Complex::new(-1.3554978623466403, -0.5155911201439718),
            Complex::new(-2.5878326142379287, 3.514608528905308),
            Complex::new(-3.3502428014666497, -2.096805136606857),
            Complex::new(2.8190562966356145, 1.1933737360686372),
            Complex::new(4.0145510535429105, 0.27487146053204814),
            Complex::new(-3.694457609076819, -4.6177275374160685),
            Complex::new(-2.9452253782796394, 0.32935476274301667),
            Complex::new(1.1201165749637167, 1.1545213950017859),
            Complex::new(-5.840730416111081, 4.033869522417064),
            Complex::new(0.5911186567532882, -2.7895298593462066),
            Complex::new(-3.591038300597631, -2.967613424416319),
            Complex::new(-2.1524574972876884, 0.36274604252619946),
            Complex::new(-1.585852957554564, 1.1499293073004022),
            Complex::new(3.2896353576728163, 2.4353728216692954),
            Complex::new(-1.9641099668008029, -1.285248043389835),
            Complex::new(-2.3778001999091787, 2.5903444593841627),
            Complex::new(-1.2978586664794591, 2.089769590603481),
            Complex::new(1.603928510376159, -2.2719394617721766),
            Complex::new(2.8868165494557356, -7.144627685991013),
            Complex::new(-1.5618096790167788, -1.080843385481133),
            Complex::new(-1.802089213465198, 2.5351142303546137),
            Complex::new(-0.745503912479854, 4.0475146316636845),
            Complex::new(2.3649902279984993, -1.3245778590379207),
            Complex::new(-6.1370536062812135, 0.9243544562224225),
            Complex::new(-4.298348593873365, 3.9414280913701676),
            Complex::new(3.084745357417178, 1.739525210776619),
            Complex::new(-1.5616325617019056, 5.076816627386519),
            Complex::new(-0.921912782425202, -3.08376970418351),
            Complex::new(-2.0671123993581335, 1.275826864128009),
            Complex::new(1.796325626659499, -1.2728959743018704),
            Complex::new(0.24956410230353798, 2.0283502602400634),
            Complex::new(2.1044071122135226, 0.10652257165607715),
            Complex::new(3.6833771026994704, -3.8205124406081525),
            Complex::new(-1.0852736998519317, -1.8177514526757572),
            Complex::new(1.7986499270880034, 0.34101541903993926),
            Complex::new(-2.2064436088357735, -4.929957991711954),
            Complex::new(-2.9054289636251593, -7.958621724893469),
            Complex::new(-0.4706705030273297, 2.3443944521019517),
            Complex::new(3.6321565747760527, 3.611500626879426),
            Complex::new(0.7696599554447048, 2.572233946615944),
            Complex::new(-2.7135073125063465, 1.7910846016088233),
            Complex::new(4.506818969482486, 0.5875255189418698),
            Complex::new(6.166753972861527, 1.5234075523408646),
            Complex::new(-1.0193655715457899, -5.250403633747775),
            Complex::new(6.626911312797217, -3.167052877690738),
            Complex::new(-1.652915074140726, -5.64926020952538),
            Complex::new(6.691495069983637, 3.49078788669754),
            Complex::new(-4.477066322208975, -6.338353113157579),
            Complex::new(0.6541416188484876, -0.08625423497295204),
            Complex::new(-10.012323033955287, -0.34956495199739335),
            Complex::new(-1.1588472759797819, -0.5280134720643622),
            Complex::new(3.6734341882027532, -4.44999591607706),
            Complex::new(-4.976183914359324, 0.8253745046854113),
            Complex::new(-1.6144455645118336, -4.504672656205258),
            Complex::new(3.878773185741312, 0.9387077336725858),
            Complex::new(-3.9448563625317266, 8.972380912176689),
            Complex::new(2.036472692860068, 0.009974226880265391),
            Complex::new(5.4602346412504525, 0.5322296358415981),
            Complex::new(-0.7208606431088669, 2.6861139970141945),
            Complex::new(-3.064253286164248, 0.010090084255546916),
            Complex::new(1.1837110753923257, 3.944325572470176),
            Complex::new(1.0697249560232707, -1.2606572469160007),
            Complex::new(0.9598966873181312, -4.735847341128473),
            Complex::new(-3.937585328943265, -0.4462065307686184),
            Complex::new(1.1977000647859606, 0.04952420072568908),
            Complex::new(-0.06965721165481692, -3.1306762326165187),
            Complex::new(6.541270226557316, 0.0),
            Complex::new(4.032159667303377, -6.074123297052789),
            Complex::new(-1.536341766311553, -1.5401315370882371),
            Complex::new(-2.033618003270432, 5.562639241397462),
            Complex::new(-0.322694934654837, 1.0726566194434715),
            Complex::new(-0.5635711289003886, 4.999853180346649),
            Complex::new(0.9036307251286286, 3.378773515530939),
            Complex::new(-1.5122675104571623, -2.0339231645341487),
            Complex::new(-1.1987118625194366, -3.3146376804251547),
            Complex::new(-0.16712742864473312, -0.1256829949494972),
            Complex::new(-2.304987777551535, 0.2660118305418626),
            Complex::new(0.7101196560489638, 4.076796913116877),
            Complex::new(-6.741126571378394, 0.0),
            Complex::new(0.7101196560489638, -4.076796913116877),
            Complex::new(-2.304987777551535, -0.2660118305418626),
            Complex::new(-0.16712742864473312, 0.1256829949494972),
            Complex::new(-1.1987118625194366, 3.3146376804251547),
            Complex::new(-1.5122675104571623, 2.0339231645341487),
            Complex::new(0.9036307251286286, -3.378773515530939),
            Complex::new(-0.5635711289003886, -4.999853180346649),
            Complex::new(-0.322694934654837, -1.0726566194434715),
            Complex::new(-2.033618003270432, -5.562639241397462),
            Complex::new(-1.536341766311553, 1.5401315370882371),
            Complex::new(4.032159667303377, 6.074123297052789),
            Complex::new(6.166753972861527, -1.5234075523408646),
            Complex::new(-0.06965721165481692, 3.1306762326165187),
            Complex::new(1.1977000647859606, -0.04952420072568908),
            Complex::new(-3.937585328943265, 0.4462065307686184),
            Complex::new(0.9598966873181312, 4.735847341128473),
            Complex::new(1.0697249560232707, 1.2606572469160007),
            Complex::new(1.1837110753923257, -3.944325572470176),
            Complex::new(-3.064253286164248, -0.010090084255546916),
            Complex::new(-0.7208606431088669, -2.6861139970141945),
            Complex::new(5.4602346412504525, -0.5322296358415981),
            Complex::new(2.036472692860068, -0.009974226880265391),
            Complex::new(-3.9448563625317266, -8.972380912176689),
            Complex::new(3.878773185741312, -0.9387077336725858),
            Complex::new(-1.6144455645118336, 4.504672656205258),
            Complex::new(-4.976183914359324, -0.8253745046854113),
            Complex::new(3.6734341882027532, 4.44999591607706),
            Complex::new(-1.1588472759797819, 0.5280134720643622),
            Complex::new(-10.012323033955287, 0.34956495199739335),
            Complex::new(0.6541416188484876, 0.08625423497295204),
            Complex::new(-4.477066322208975, 6.338353113157579),
            Complex::new(6.691495069983637, -3.49078788669754),
            Complex::new(-1.652915074140726, 5.64926020952538),
            Complex::new(6.626911312797217, 3.167052877690738),
            Complex::new(-1.0193655715457899, 5.250403633747775),
            Complex::new(2.8868165494557356, 7.144627685991013),
            Complex::new(4.506818969482486, -0.5875255189418698),
            Complex::new(-2.7135073125063465, -1.7910846016088233),
            Complex::new(0.7696599554447048, -2.572233946615944),
            Complex::new(3.6321565747760527, -3.611500626879426),
            Complex::new(-0.4706705030273297, -2.3443944521019517),
            Complex::new(-2.9054289636251593, 7.958621724893469),
            Complex::new(-2.2064436088357735, 4.929957991711954),
            Complex::new(1.7986499270880034, -0.34101541903993926),
            Complex::new(-1.0852736998519317, 1.8177514526757572),
            Complex::new(3.6833771026994704, 3.8205124406081525),
            Complex::new(2.1044071122135226, -0.10652257165607715),
            Complex::new(0.24956410230353798, -2.0283502602400634),
            Complex::new(1.796325626659499, 1.2728959743018704),
            Complex::new(-2.0671123993581335, -1.275826864128009),
            Complex::new(-0.921912782425202, 3.08376970418351),
            Complex::new(-1.5616325617019056, -5.076816627386519),
            Complex::new(3.084745357417178, -1.739525210776619),
            Complex::new(-4.298348593873365, -3.9414280913701676),
            Complex::new(-6.1370536062812135, -0.9243544562224225),
            Complex::new(2.3649902279984993, 1.3245778590379207),
            Complex::new(-0.745503912479854, -4.0475146316636845),
            Complex::new(-1.802089213465198, -2.5351142303546137),
            Complex::new(-1.5618096790167788, 1.080843385481133),
            Complex::new(3.9380430471136094, 2.2336101756446247),
            Complex::new(1.603928510376159, 2.2719394617721766),
            Complex::new(-1.2978586664794591, -2.089769590603481),
            Complex::new(-2.3778001999091787, -2.5903444593841627),
            Complex::new(-1.9641099668008029, 1.285248043389835),
            Complex::new(3.2896353576728163, -2.4353728216692954),
            Complex::new(-1.585852957554564, -1.1499293073004022),
            Complex::new(-2.1524574972876884, -0.36274604252619946),
            Complex::new(-3.591038300597631, 2.967613424416319),
            Complex::new(0.5911186567532882, 2.7895298593462066),
            Complex::new(-5.840730416111081, -4.033869522417064),
            Complex::new(1.1201165749637167, -1.1545213950017859),
            Complex::new(-2.9452253782796394, -0.32935476274301667),
            Complex::new(-3.694457609076819, 4.6177275374160685),
            Complex::new(4.0145510535429105, -0.27487146053204814),
            Complex::new(2.8190562966356145, -1.1933737360686372),
            Complex::new(-3.3502428014666497, 2.096805136606857),
            Complex::new(-2.5878326142379287, -3.514608528905308),
            Complex::new(-1.3554978623466403, 0.5155911201439718),
            Complex::new(-1.79799016628888, -11.814947622610552),
            Complex::new(-0.7553982110067918, 4.521577531461968),
            Complex::new(-3.8248062027390475, 2.5250116691631277),
            Complex::new(-1.7813777739001353, -11.556784291451603),
            Complex::new(6.080940845573007, 2.324482277023904),
            Complex::new(2.573481318982692, 2.542591689870939),
            Complex::new(0.51222948521139, 3.0422742756607963),
            Complex::new(8.61082169972553, -4.801900847453371),
            Complex::new(-2.343176995718726, 4.8101493001419975),
            Complex::new(2.071398814561847, 2.4500203908496303),
            Complex::new(-3.53788359364677, -1.9680763523721678),
            Complex::new(0.7673491433916113, -1.8345734779370055),
            Complex::new(2.6252451595071933, -0.6551883981264515),
            Complex::new(-0.8057950339464541, 2.028750060451022),
            Complex::new(-4.499470544262742, -5.785691094584783),
            Complex::new(-0.8401788927293691, -3.409461120256584),
            Complex::new(3.26214566690222, -4.4555909540179455),
            Complex::new(2.612825067014143, 0.6262449302657651),
            Complex::new(5.195168946325346, 5.599599699993183),
            Complex::new(-4.651734564401249, 3.957622481137731),
            Complex::new(0.9431463653346317, -1.4900894364351855),
            Complex::new(-0.4272378426861331, -2.472616834988732),
            Complex::new(0.2548853901484276, -6.252250833184429),
            Complex::new(-8.426737895239928, -1.477059762741725),
            Complex::new(-1.8693468194234835, -0.46998827429938417),
            Complex::new(3.4396356375736, 0.4342370304384102),
            Complex::new(-0.8454581277394115, 0.7596011837107017),
            Complex::new(9.200340526131951, -3.674620173296757),
            Complex::new(-0.2608678824289141, -4.038843854571754),
            Complex::new(-1.3717992651530322, -0.4889251692928045),
            Complex::new(1.0390903665791136, 0.4092004505236395),
            Complex::new(-1.8085495290827898, -3.5789274626261838),
            Complex::new(-5.6096175895169775, 2.307740061532588),
            Complex::new(3.789956544025782, -0.8089187417597645),
            Complex::new(-0.7055687959474743, -3.593264353832105),
            Complex::new(-2.0834510892728146, 4.154462997012521),
            Complex::new(-1.6584579884976751, 6.310483857378658),
            Complex::new(2.3682628894254423, -2.23008329676306),
            Complex::new(2.824412135830933, 5.962685818348512),
            Complex::new(4.237273462279213, 4.143807763539963),
            Complex::new(0.8776658356175657, 2.065150780332987),
            Complex::new(3.1443637804160414, 2.1820688693695383),
            Complex::new(2.3541456077858585, -1.4331453706786732),
            Complex::new(1.9130057483669187, -0.13512884581887286),
            Complex::new(2.467673303314676, -5.845782635321265),
            Complex::new(4.115157876324687, -2.2409594223568083),
            Complex::new(2.908989148177296, 5.1175738846889445),
            Complex::new(1.8517301932817827, -2.392592405787043),
            Complex::new(7.821013356793824, 4.828722008992704),
            Complex::new(2.321351756601554, 5.779177535488895),
            Complex::new(5.7353056258742745, 0.3140966750394031),
            Complex::new(-0.631000565802627, -1.2481053818696088),
            Complex::new(0.22704221699843447, 1.1641261321031933),
        ];
        let sim = build_test_sim();
        let mut in_fld = Field::new(&sim);
        let mut fft_2d = Fft2D::new(&sim);
        assert_eq!(in_fld.spectral.len(), input.len());
        in_fld.spectral = input;
        fft_2d.fft(&mut in_fld);
        assert_eq!(in_fld.spectral.len(), out.len());
        for (v1, v2) in in_fld.spectral.iter().zip(out) {
            // assert_eq!(v1.re, v2.re);
            assert!((v1.re - v2.re) < E_TOL);
            assert!((v1.im - v2.im) < E_TOL);
        }
    }

    #[test]
    fn transpose() {
        let mut input = get_2d_input();
        let sim = build_test_sim();
        let mut fft_2d = Fft2D::new(&sim);
        Fft2D::transpose_out_of_place(&mut input, &mut fft_2d.wrkspace, &mut fft_2d.field_size);
        let transpose_dim = FieldDim {
            size_x: fft_2d.field_size.size_y,
            size_y: fft_2d.field_size.size_x,
        };

        for i in 0..fft_2d.field_size.size_y {
            for j in 0..fft_2d.field_size.size_x {
                let in_ind = transpose_dim.get_index(Pos { row: j, col: i });
                let out_ind = fft_2d.field_size.get_index(Pos { row: i, col: j });
                assert_eq!(input[in_ind], fft_2d.wrkspace[out_ind]);
            }
        }
    }

    #[test]
    fn fft_2d_roundtrip() {
        let input = get_2d_input();
        let sim = build_test_sim();
        let mut in_fld = Field::new(&sim);
        let mut fft_2d = Fft2D::new(&sim);
        assert_eq!(in_fld.spectral.len(), input.len());
        in_fld.spectral = get_2d_input();
        fft_2d.fft(&mut in_fld);
        // check that the value has changed
        assert!(in_fld.spectral.iter().zip(input.iter()).any(|(v1, v2)| {
            // assert_eq!(v1.re, v2.re);
            ((v1.re - v2.re) > E_TOL) || ((v1.im - v2.im) > E_TOL)
        }));

        fft_2d.inv_fft(&mut in_fld);

        assert_eq!(in_fld.spectral.len(), input.len());
        for (v1, v2) in in_fld.spectral.iter().zip(input.iter()) {
            // assert_eq!(v1.re, v2.re);
            assert!((v1.re - v2.re) < E_TOL);
            assert!((v1.im - v2.im) < E_TOL);
        }
    }
}
