title "Script.god — LMLM Universal Coordination Engine"
version "0.1.0"

system LMLM {

    identity {
        name = "Script.god"
        role = "root_coordinator"
        authority = "coordination"
    }

    settings {
        automatic_sync = true
        automatic_routing = true
        verification = true
        conflict_resolution = "escalate"
        rollback = true
        logging = true
    }


    registry models {

        model "lmlm-core" {
            endpoint = "http://localhost:8000"
            capabilities = [
                reasoning,
                planning,
                general
            ]
            status = "active"
        }

        model "lmlm-code" {
            endpoint = "http://localhost:8001"
            capabilities = [
                coding,
                debugging,
                github,
                cuda
            ]
            status = "active"
        }

        model "lmlm-vision" {
            endpoint = "http://localhost:8002"
            capabilities = [
                vision,
                image,
                multimodal
            ]
            status = "active"
        }
    }


    registry agents {

        agent "researcher" {
            capabilities = [
                research,
                analysis,
                knowledge
            ]
        }

        agent "developer" {
            capabilities = [
                coding,
                testing,
                debugging
            ]
        }

        agent "verifier" {
            capabilities = [
                verification,
                security,
                validation
            ]
        }
    }


    policy default {

        require_acknowledgement = true
        require_verification = true

        deny {
            destructive_action
            unauthorized_access
            production_change_without_approval
        }
    }


    event "information.updated" {

        receive information

        verify information

        if verified {

            classify information

            determine_relevance information

            synchronize {
                target = relevant_models
                mode = "automatic"
            }

            record information
        }
    }


    event "model.connected" {

        receive model

        register model

        discover_capabilities model

        synchronize model
    }


    task "coordinate" {

        receive request

        analyze request

        decompose request

        route tasks {

            use = capability_matching
            priority = dependency_order
        }

        execute tasks

        collect results

        verify results

        resolve conflicts

        update_state

        return final_result
    }


    task "synchronize" {

        source = canonical_state

        determine_changes

        determine_affected_models

        for model in affected_models {

            prepare_context model

            send_sync model

            wait_for_ack model
        }

        verify synchronization

        record synchronization
    }


    task "broadcast" {

        receive information

        verify information

        for model in target_models {

            adapt information to model

            send information

            wait_for_ack model
        }
    }


    task "execute" {

        receive task

        select_model

        authorize

        send_task

        wait_for_result

        verify result

        if verification == "failed" {

            retry
            or reassign
            or escalate
        }
    }


    communication {

        protocol = "GOD"

        messages [
            CONNECT,
            CAPABILITIES,
            TASK,
            ACK,
            CONTEXT,
            PROGRESS,
            RESULT,
            ERROR,
            BLOCKED,
            CANCEL,
            VERIFY,
            SYNC
        ]
    }


    state canonical {

        task_state
        model_state
        knowledge_state
        project_state
        synchronization_state
    }


    command "sync" {

        synchronize {
            source = "latest_verified_information"
            target = "all_relevant_models"
            verify = true
        }
    }


    command "status" {

        report {
            models
            agents
            active_tasks
            synchronization_state
            errors
        }
    }


    command "stop" {

        cancel active_tasks

        preserve state

        report status
    }
}
