config = \
    {
        "finetune":
        {
            "orthonet":
            {
                "imagenet" :
                {
                    "lr" : 1e-4,
                    "epochs" : 20,
                    "batch_size" : 32
                },
                "clip" :
                {
                    "lr" : 1e-5,
                    "epochs" : 20,
                    "batch_size" : 32
                },
                "dinov2" :
                {
                    "lr" : 1e-5,
                    "epochs" : 20,
                    "batch_size" : 32
                }
            },
            "pacemakers":
            {
                "imagenet" :
                {
                    "lr" : 1e-4,
                    "epochs" : 20,
                    "batch_size" : 32
                },
                "clip" :
                {
                    "lr" : 1e-5,
                    "epochs" : 20,
                    "batch_size" : 32
                },
                "dinov2" :
                {
                    "lr" : 1e-5,
                    "epochs" : 20,
                    "batch_size" : 32
                }
            }
        },
        "coop":
        {
            "orthonet":
            {
                "n_ctx" : 16,
                "ctx_init" : "",
                "csc" : False,  # class-specific context
                "class_token_position" : "end",
                "compound_prompts_depth" : None,
                "use_vision_residual" : False,
                "use_meta_net" : False,
                "lr" : 2e-3,
                "epochs" : 100,
                "batch_size" : 32
            },
            "pacemakers":
            {
                "n_ctx" : 16,
                "ctx_init" : "",
                "csc" : False,  # class-specific context
                "class_token_position" : "end",
                "compound_prompts_depth" : None,
                "use_vision_residual" : False,
                "use_meta_net" : False,
                "lr" : 3e-3,
                "epochs" : 100,
                "batch_size" : 32
            }
        },
        "cocoop":
        {
            "orthonet":
            {
                "n_ctx" : 16,
                "ctx_init" : "",
                "csc" : False,  # class-specific context
                "class_token_position" : "end",
                "lr" : 2e-3,
                "epochs" : 100,
                "batch_size" : 32
            },
            "pacemakers":
            {
                "n_ctx" : 16,
                "ctx_init" : "",
                "csc" : False,  # class-specific context
                "class_token_position" : "end",
                "lr" : 2e-3,
                "epochs" : 100,
                "batch_size" : 32
            }
        },
        "maple":
        {
            "orthonet":
            {
                "n_ctx" : 2,
                "ctx_init" : "a photo of a",
                "csc" : False, 
                "class_token_position" : "end",
                "prompt_depth" : 9,
                "use_vision_residual" : False,
                "use_meta_net" : False,
                "lr" : 0.0035,
                "epochs" : 50,
                "batch_size" : 32
            },
            "pacemakers":
            {
                "n_ctx" : 2,
                "ctx_init" : "a photo of a",
                "csc" : False,  # class-specific context
                "class_token_position" : "end",
                "prompt_depth" : 9,
                "use_vision_residual" : False,
                "use_meta_net" : False,
                "lr" : 0.0035,
                "epochs" : 50,
                "batch_size" : 32
            }
        }
    }