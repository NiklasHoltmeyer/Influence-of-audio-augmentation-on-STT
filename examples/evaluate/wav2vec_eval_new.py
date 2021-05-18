from tqdm import tqdm
nh_proc="test"
jobs = [
#    ('/share/modelle/no_aug/models/cv_small_noaug_2/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_2/checkpoint-4000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_2/checkpoint-2000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-4000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-5000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-8000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-7000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-6000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1_conf/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1_conf/checkpoint-4000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1_conf/checkpoint-2000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_s_1_conf/checkpoint-6000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1/checkpoint-4000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1/checkpoint-2000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-4000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-10000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-5000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-11000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-8000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-2000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-7000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-6000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-9000', nh_proc),
    ('/share/modelle/no_aug/models/cv_s_1/checkpoint-6000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-3000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-4000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-10000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-5000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-8000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-2000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-7000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-6000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-9000', nh_proc),
    ('/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-18000',
     '/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-18000'),
    ('/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-19000',
     '/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-19000'),
    ('/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-17000',
     '/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-17000'),
    (
    '/share/modelle/base/models/run_g_f_i_1/checkpoint-3000', '/share/modelle/base/models/run_g_f_i_1/checkpoint-3000'),
    (
    '/share/modelle/base/models/run_g_f_i_1/checkpoint-1000', '/share/modelle/base/models/run_g_f_i_1/checkpoint-1000'),
    (
    '/share/modelle/base/models/run_g_f_i_1/checkpoint-2000', '/share/modelle/base/models/run_g_f_i_1/checkpoint-2000'),
    ('/share/modelle/base/models/run_k_3/checkpoint-5000', '/share/modelle/base/models/run_k_3/checkpoint-5000'),
    ('/share/modelle/base/models/run_k_3/checkpoint-1000', '/share/modelle/base/models/run_k_3/checkpoint-1000'),
    ('/share/modelle/base/models/run_k_3/checkpoint-2000', '/share/modelle/base/models/run_k_3/checkpoint-2000'),
    ('/share/modelle/base/models/run_pro_idleback/checkpoint-22000',
     '/share/modelle/base/models/run_pro_idleback/checkpoint-22000'),
    ('/share/modelle/base/models/run_pro_idleback/checkpoint-23000',
     '/share/modelle/base/models/run_pro_idleback/checkpoint-23000'),
    ('/share/modelle/base/models/run_pro_idleback/checkpoint-24000',
     '/share/modelle/base/models/run_pro_idleback/checkpoint-24000'),
    ('/share/modelle/base/models/run_k_1/checkpoint-3000', '/share/modelle/base/models/run_k_1/checkpoint-3000'),
    ('/share/modelle/base/models/run_k_1/checkpoint-4000', '/share/modelle/base/models/run_k_1/checkpoint-4000'),
    ('/share/modelle/base/models/run_k_1/checkpoint-5000', '/share/modelle/base/models/run_k_1/checkpoint-5000'),
    ('/share/modelle/base/models/run_pro_750_wu/checkpoint-8000',
     '/share/modelle/base/models/run_pro_750_wu/checkpoint-8000'),
    ('/share/modelle/base/models/run_pro_750_wu/checkpoint-1000',
     '/share/modelle/base/models/run_pro_750_wu/checkpoint-1000'),
    ('/share/modelle/base/models/run_pro_750_wu/checkpoint-9000',
     '/share/modelle/base/models/run_pro_750_wu/checkpoint-9000'),
    ('/share/modelle/base/models/run_pro_500_wu/checkpoint-8000',
     '/share/modelle/base/models/run_pro_500_wu/checkpoint-8000'),
    ('/share/modelle/base/models/run_pro_500_wu/checkpoint-7000',
     '/share/modelle/base/models/run_pro_500_wu/checkpoint-7000'),
    ('/share/modelle/base/models/run_pro_500_wu/checkpoint-9000', '/share/modelle/base/models/run_pro_500_wu/checkpoint - 9000'),
]

def already_run(model_name, file_path):
    with open(file_path, "r+") as f:
        for line in f.readlines():
            line_model = line.split("\t")[0]
            if model_name in line_model:
                return True
    return False


for model_name, based_on in tqdm(
        jobs, desc="troll"):
    try:
        # base_on = str(Path(model_name).parent.resolve())
        # base_on = "/share/datasets/wav2vec2-large-xlsr-german-vf_nh"
        if not already_run(model_name, "/share/modelle/check_results_checkpoints_bruteforce.tsv"):
            result_tsv = validate_model(model_name, based_on=based_on)

            with open("/share/modelle/check_results_checkpoints_bruteforce.tsv", "a+") as f:
                f.write("\t".join([result_tsv]))
                f.write("\n")

            results.append(result_tsv)
    except Exception as e:
        print(e)
        with open("/share/modelle/check_failed.tsv", "a+") as f:
            f.write("\t".join([model_name, "2" + str(e)]))
            f.write("\n")
        failed.append(model_name)