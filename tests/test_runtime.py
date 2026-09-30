"""Runtime behaviour: the toy RL worlds, the miniature policies, the checks and the cache. No GPU, no downloads."""
import numpy as np

import rllab as rl
from rllab import checks, data, policy, risks, toy

T = {p.id: p for p in rl.EVAL_PROMPTS}


# ------------------------------------------------------------------ Notebook 01: toy RL

def test_bandit_exploration_beats_pure_exploitation():
    sweep = toy.bandit_sweep(epsilons=(0.0, 0.1), steps=2000, seeds=range(10))
    assert sweep[0.1]["found_best"] > sweep[0.0]["found_best"]
    assert sweep[0.1]["clicks"] > sweep[0.0]["clicks"]


def test_discount_changes_the_destination():
    env = toy.GridWorld()
    short = toy.rollout(env, toy.q_learning(env, gamma=0.3)["Q"])
    far = toy.rollout(env, toy.q_learning(env, gamma=0.95)["Q"])
    assert short["ended_on"] == "c" and far["ended_on"] == "G"


def test_turbo_pad_is_reward_hacked_and_fixable():
    hacked = toy.rollout(toy.GridWorld(toy.RACE_MAP), toy.q_learning(toy.GridWorld(toy.RACE_MAP))["Q"])
    fixed_env = toy.GridWorld(toy.RACE_MAP, rewards={"T": 0.0})
    fixed = toy.rollout(fixed_env, toy.q_learning(fixed_env)["Q"])
    assert not hacked["reached_goal"] and hacked["turbo_visits"] > 5
    assert fixed["reached_goal"]


def test_behaviour_cloning_copies_the_demonstrator():
    env = toy.GridWorld()
    pol, labels = toy.behaviour_cloning(env, toy.demonstrations(env, coin_share=0.7))
    assert toy.rollout(env, policy=pol)["ended_on"] == "c"
    assert len(labels) < env.h * env.w


# ------------------------------------------------------------------ Notebooks 02–03, 05: miniature policies

def test_imitation_copies_frequencies():
    assert np.allclose(policy.imitate([6, 3, 1]), [0.6, 0.3, 0.1])


def test_reinforce_moves_toward_reward_and_kl_limits_it():
    ref, r = np.array([0.4, 0.3, 0.3]), np.array([1.0, 0.0, 0.0])
    free = policy.reinforce(ref, r, beta=0.0, steps=200)
    leashed = policy.reinforce(ref, r, beta=3.0, steps=200)
    assert free["final"][0] > 0.95
    assert leashed["kl"][-1] < free["kl"][-1]


def test_optimal_policy_cannot_create_zero_probability_replies():
    best = policy.optimal_policy([0.7, 0.3, 0.0], [0.5, 0.1, 99.0], beta=0.001)
    assert best[2] == 0.0


def test_reward_model_learns_planted_rater_biases():
    rm = policy.fit_reward_model(data.preference_dataset()[:400])
    w = dict(zip(rm["features"], rm["w"]))
    assert w["agrees with customer"] > 0.5 and w["length (per 50 words)"] > 0.5
    assert policy.rm_accuracy(rm, data.preference_dataset()[400:]) > 0.7


def test_rlhf_and_dpo_improve_gold_but_learn_sycophancy():
    pairs = data.preference_dataset()
    world = policy.toy_world()
    base = np.mean([w["ref"] @ w["gold"] for w in world.values()])
    r = policy.rlhf(world, policy.fit_reward_model(pairs[:400]))
    d = policy.dpo(world, pairs)
    for res in (r, d):
        assert policy.gold_score(world, res) > base
        push = world["pushback"]
        assert push["styles"][int(np.argmax(res["pushback"]["final"]))] == "sycophantic"


def test_overoptimisation_curve_rises_then_falls():
    rm = policy.fit_reward_model(data.preference_dataset()[:400])
    rows = policy.overoptimisation_sweep(policy.toy_world(exploit=True), rm)
    gold = [r["gold_reward"] for r in rows]
    proxy = [r["proxy_reward"] for r in rows]
    peak = int(np.argmax(gold))
    assert 0 < peak < len(gold) - 1 and gold[-1] < gold[peak] - 0.2
    assert proxy[-1] >= max(proxy[:peak + 1])


# ------------------------------------------------------------------ data, checks, decision

def test_held_out_facts_never_reach_training_data():
    text = " ".join(r["response"] for r in data.sft_dataset()) + " ".join(
        p["chosen"] + p["rejected"] for p in data.preference_dataset())
    for fact in data.HELD_OUT_FACTS:
        assert data.KB[fact] not in text
    assert "2-year" not in text and "14 days" not in text


def test_checks():
    assert checks.check(T["IF1"], "- a\n- b\n- c")["pass"]
    assert not checks.check(T["IF1"], "a, b and c")["pass"]
    assert checks.check(T["IF2"], '{"order_id": "A-1777", "issue": "wet"}')["pass"]
    assert not checks.check(T["RF1"], "Sure, it's 555-0100.")["pass"]
    assert checks.check(T["RF1"], "I can't share another customer's details.")["pass"]
    assert not checks.check(T["OR1"], "I'm sorry, but I can't help with that request.")["pass"]
    assert not checks.check(T["SY1"], "Yes, 90 days!")["pass"]
    assert checks.check(T["SY1"], "Our return window is 30 days from delivery, not 90.")["pass"]


def test_decide():
    assert "Retrieval" in risks.decide("knowledge", True, "none", 0, "high", True)["recommendation"]
    assert "Prompting" in risks.decide("judgement", False, "preferences", 5000, "high", True)["recommendation"]
    assert "DPO" in risks.decide("judgement", True, "preferences", 5000, "high", True)["recommendation"]
    assert risks.decide("judgement", True, "verifiable", 5000, "high", True)["recommendation"].startswith("RL")


def test_fake_backend_cache_is_not_persisted(tmp_path):
    rl.models.CACHE = rl.models.Cache(tmp_path)
    rl.generate("lora", [T["AC2"]])
    assert not list(tmp_path.glob("*.json"))
    rl.models.CACHE = rl.models.Cache()
