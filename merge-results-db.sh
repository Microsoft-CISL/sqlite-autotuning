#!/bin/bash

set -eu

source_db="mlos_bench.sqlite.bak"
target_db="mlos_bench.sqlite"

# objectives

set -x

cat <<EOF | sqlite3 -echo
ATTACH DATABASE '$source_db' AS source_db;
ATTACH DATABASE '$target_db' AS target_db;

BEGIN TRANSACTION;

PRAGMA defer_foreign_keys = true;

CREATE TEMPORARY TABLE source_experiments AS
SELECT exp_id FROM source_db.experiment
WHERE exp_id NOT IN (SELECT exp_id FROM target_db.experiment)
;

-- ALTER TABLE target_db.experiment DROP COLUMN optimization_target;
-- ALTER TABLE target_db.experiment DROP COLUMN optimization_direction;

INSERT INTO target_db.experiment (exp_id, description, root_env_config, git_repo, git_commit)
SELECT exp_id, description, root_env_config, git_repo, git_commit FROM source_db.experiment
WHERE exp_id IN (SELECT exp_id FROM source_experiments)
;

/*
-- Increment the config_ids in source_db so that they don't overlap the target_db configs.
-- NOTE: This currently assumes the config_hashes are still unique.
-- FIXME: Non-idempotent.
CREATE TEMPORARY TABLE max_config_id AS
SELECT MAX(
    (SELECT MAX(config_id) FROM source_db.config),
    (SELECT MAX(config_id) FROM target_db.config)
) AS config_id;
UPDATE source_db.config SET config_id = config_id + (SELECT MAX(config_id) FROM max_config_id);
UPDATE source_db.config_param SET config_id = config_id + (SELECT MAX(config_id) FROM max_config_id);
UPDATE source_db.trial SET config_id = config_id + (SELECT MAX(config_id) FROM max_config_id);
*/

INSERT INTO target_db.config (config_id, config_hash)
SELECT config_id, config_hash FROM source_db.config
WHERE config_id NOT IN (SELECT config_id FROM target_db.config)
AND config_hash NOT IN (SELECT config_hash FROM target_db.config)
;

INSERT INTO target_db.config_param (config_id, param_id, param_value)
SELECT config_id, param_id, param_value FROM source_db.config_param
WHERE config_id NOT IN (SELECT config_id FROM target_db.config_param)
;

INSERT INTO target_db.trial (exp_id, trial_id, config_id, ts_start, ts_end, status)
SELECT exp_id, trial_id, config_id, ts_start, ts_end, status FROM source_db.trial
WHERE exp_id IN (SELECT exp_id FROM source_experiments)
;

INSERT INTO target_db.trial_param (exp_id, trial_id, param_id, param_value)
SELECT exp_id, trial_id, param_id, param_value FROM source_db.trial_param
WHERE exp_id IN (SELECT exp_id FROM source_experiments)
;

INSERT INTO target_db.trial_result (exp_id, trial_id, metric_id, metric_value)
SELECT exp_id, trial_id, metric_id, metric_value FROM source_db.trial_result
WHERE exp_id IN (SELECT exp_id FROM source_experiments)
;

INSERT INTO target_db.trial_telemetry (exp_id, trial_id, ts, metric_id, metric_value)
SELECT exp_id, trial_id, ts, metric_id, metric_value FROM source_db.trial_telemetry
WHERE exp_id IN (SELECT exp_id FROM source_experiments)
;

SELECT * FROM target_db.experiment;
/*
SELECT * FROM target_db.config;
SELECT * FROM target_db.config_param;
SELECT * FROM target_db.trial;
SELECT * FROM target_db.trial_param;
SELECT * FROM target_db.trial_result;
SELECT * FROM target_db.trial_telemetry;
*/

ROLLBACK TRANSACTION;
-- COMMIT TRANSACTION;
EOF
