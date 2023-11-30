

MODEL_NAMES = {
    "roberta_L6_H768": "nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large",
    "roberta_L6_H384": "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large",
    "xlmr_L12_H384": "nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large",
    "xlmr_L6_H384": "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large",
}

rule all:
    input:
        #expand('clustered_models/{base_model}_{vocab}k/done', base_model=MODEL_NAMES.keys(), vocab=[1, 10, 100, 1000, 10000])
        expand('mnli_models/{base_model}_{vocab}k/done',
               base_model=["roberta_L6_H768", "roberta_L6_H384"],
               vocab=[5, 10, 20, 30, 40])

rule cluster_embedding:
    output:
        out_dir=directory('clustered_models/{base_model}_{vocab}k'),
        done='clustered_models/{base_model}_{vocab}k/done'
    params:
        hf_model=lambda wildcards: MODEL_NAMES[wildcards.base_model],
    resources:
        mem="40G",
        cpus_per_task=16,
    shell:
        '''
        source env/bin/activate
        python3 cluster_embeddings.py {params.hf_model} {wildcards.vocab}000 {output.out_dir}
        touch {output}
        '''

rule mnli_with_clustered:
    input:
        'clustered_models/{base_model}_{vocab}k/done'
    output:
        "mnli_models/{base_model}_{vocab}k/done"
    resources:
        mem="30G",
        cpus_per_task=4,
        partition="gpu-troja,gpu-ms",
        flags="--gres=gpu:1"
    shell:
        '''
        source env/bin/activate
        python3 run_glue.py \
            --model_name_or_path clustered_models/{wildcards.base_model}_{wildcards.vocab}k \
            --task_name mnli \
            --output_dir mnli_models/{wildcards.base_model}_{wildcards.vocab}k \
            --do_train \
            --do_eval \
            --max_seq_length 256 \
            --per_device_train_batch_size 64 \
            --learning_rate 2e-5 \
            --num_train_epochs 3
        touch {output}
        '''

rule mnli_with_base:
    output:
        "mnli_models/{base_model}_base/done"
    params:
        hf_model=lambda wildcards: MODEL_NAMES[wildcards.base_model],
    resources:
        mem="30G",
        cpus_per_task=4,
        partition="gpu-troja,gpu-ms",
        flags="--gres=gpu:1"
    shell:
        '''
        source env/bin/activate
        python3 run_glue.py \
            --model_name_or_path {params.hf_model} \
            --task_name mnli \
            --output_dir mnli_models/{wildcards.base_model} \
            --do_train \
            --do_eval \
            --max_seq_length 256 \
            --per_device_train_batch_size 64 \
            --learning_rate 2e-5 \
            --num_train_epochs 3
        touch {output}
        '''


#rule squad_with_clustered:
#    pass


#rule squad_with_base:
#    pass


# TODO prepare ours subword embeddings
# 1. pre-tokenize corpus
# 2. train fasttext on the pretokenized corpus
# 3. tokenize vocabulary with HF model tokenizer
# 4. run our tool for 1 epoch
