function format_prob(prob, n_samples, decimal_places=1) {
  if (prob < 1/n_samples)
      return "-";
  else if (prob > (n_samples - 1)/n_samples)
      return "✓"
  else if (prob < 0.001)
      return "<0.1%"
  else if (prob > 0.999)
      return ">99.9%"
  else
      return (prob*100).toFixed(decimal_places) + "%"
}

function color_prob(prob, color) {
  if (color === "upper" || color === "playoff" || color === "promotion") {
    return "rgb(" + Math.round(235 - 200*prob) + "," + Math.round(255 - 90*prob)
            + "," + Math.round(235 - 170*prob) + ")";
  }
  else if (color === "gs") {
    return "rgb(" + Math.round(245 - 120*prob) + "," + Math.round(245 - 50*prob)
            + "," + Math.round(215 - 215*prob) + ")";
  }
  else if (color === "lower" || color === "wildcard") {
    return "rgb(" + Math.round(255 - 40*prob) + "," + Math.round(230 - 145*prob)
            + "," + Math.round(205 - 200*prob) + ")";
  }
  else if (color === "relegation" || color === "elim") {
    return "rgb(" + Math.round(255 - 30*prob) + "," + Math.round(230 - 185*prob)
            + "," + Math.round(225 - 170*prob) + ")";
  }
  else if (color === "green") {
    return "rgb(" + Math.round(240 - 140*prob) + "," + Math.round(240 - 70*prob)
            + "," + Math.round(240 - 140*prob) + ")";
  }
  else
    return "rgb(" + Math.round(240 - 120*prob) + "," + Math.round(240 - 130*prob)
            + "," + Math.round(240 - 70*prob) + ")";
}

function render_team_probs_dpc(team_data, team_index, group,
                               wildcard_slots, n_samples) {
  var prob_types = {
    "upper": ["playoff", "gs", "wildcard", "relegation"],
    "lower": ["promotion", "relegation"]};
  for (let type_i = 0; type_i < prob_types[group].length; type_i++) {
    let prob_type = prob_types[group][type_i];
    if (prob_type === "wildcard" && wildcard_slots == 0)
      continue;

    let prob_html = document.getElementById("gs-" + prob_type + "-prob-" + group + "-" + team_index);
    let prob = 0;
    if (prob_type == "playoff")
      prob = team_data["probs"][0];
    else if (prob_type == "promotion")
      prob = team_data["probs"][0] + team_data["probs"][1];
    else if (prob_type == "gs")
      prob = team_data["probs"][1];
    else if (prob_type == "wildcard") {
      for (let slot = 0; slot < wildcard_slots; slot++)
      prob += team_data["probs"][2 + slot];
    }
    else
      prob = team_data["probs"][6] + team_data["probs"][7];
    prob_html.innerText = format_prob(prob, n_samples);
    prob_html.style.backgroundColor = color_prob(prob, prob_type);
  }
}

function render_team_probs_ti(team_data, team_index, group, n_samples) {
  var prob_types = ["upper", "lower", "elim"];
  for (let type_i = 0; type_i < 3; type_i++) {
    let prob_type = prob_types[type_i];
    let prob_html = document.getElementById("gs-" + prob_type + "-prob-" + group + "-" + team_index);
    let prob = 0;
    if (prob_type == "upper")
      prob = team_data["probs"].slice(0, 4).reduce((a, b) => a + b, 0);
    else if (prob_type == "lower")
      prob = team_data["probs"].slice(4, 8).reduce((a, b) => a + b, 0);
    else
      prob = team_data["probs"][8];
    prob_html.innerText = format_prob(prob, n_samples);
    prob_html.style.backgroundColor = color_prob(prob, prob_type);
  }
}

function render_team_probs_major(team_data, team_index, group, n_samples) {
  if (group === "gs") {
    var prob_types = ["upper", "lower", "elim"];
    for (let type_i = 0; type_i < 3; type_i++) {
      let prob_type = prob_types[type_i];
      let prob_html = document.getElementById("gs-" + prob_type + "-prob-" + group + "-" + team_index);
      let prob = 0;
      if (prob_type == "upper")
        prob = team_data["probs"].slice(0, 2).reduce((a, b) => a + b, 0);
      else if (prob_type == "lower")
        prob = team_data["probs"].slice(2, 6).reduce((a, b) => a + b, 0);
      else
        prob = team_data["probs"].slice(6, 8).reduce((a, b) => a + b, 0);
      prob_html.innerText = format_prob(prob, n_samples);
      prob_html.style.backgroundColor = color_prob(prob, prob_type);
    }
  }
  else {
    var prob_types = ["gs", "elim"];
    for (let type_i = 0; type_i < 2; type_i++) {
      let prob_type = prob_types[type_i];
      let prob_html = document.getElementById("gs-" + prob_type + "-prob-" + group + "-" + team_index);
      let prob = 0;
      if (prob_type == "gs")
        prob = team_data["probs"].slice(0, 2).reduce((a, b) => a + b, 0);
      else
        prob = team_data["probs"].slice(2, 6).reduce((a, b) => a + b, 0);
      prob_html.innerText = format_prob(prob, n_samples);

      if (prob_type === "gs")
        prob_html.style.backgroundColor = color_prob(prob, "upper");
      else
        prob_html.style.backgroundColor = color_prob(prob, prob_type);
    }
  }
}

function render_group_rank_probs(format, sim_data, wildcard_slots) {
  if (format === "ti")
    var groups = ['a', 'b'];
  else if (format === "dpc")
    var groups = ['upper', 'lower'];
  else if (format === "major")
    var groups = ['wc', 'gs'];

  for (let group_i = 0; group_i < 2; group_i++) {
    let group = groups[group_i];
    let team_count = sim_data["probs"]["group_rank"][group].length;
    for (let i = 0; i < team_count; i++) {
      let team_data = sim_data["probs"]["group_rank"][group][i]
      let team = team_data["team"];

      document.getElementById("gs-team-" + group + "-" + i).innerText = team;
      document.getElementById("gs-im-" + group + "-" + i).src = "image/"+team+".png";
      document.getElementById("gs-rating-" + group + "-" + i).innerText = sim_data["ratings"][team];
      if (format === "ti") {
        let record = sim_data["records"][team];
        document.getElementById("gs-record-" + group + "-" + i).innerText = (
          record[0] + "-" + record[1] + "-" + record[2]);
      }
      else if (format === "dpc") {
        let record = sim_data["records"][team];
        document.getElementById("gs-record-" + group + "-" + i).innerText = (
          record[0] + "-" + record[1]);
      }
      else if (format === "major") {
        let record = [0, 0, 0];
        if (group === "wc")
          record = sim_data["records"]["wildcard"][team];
        else if (group === "gs")
          record = sim_data["records"]["group stage"][team];
        document.getElementById("gs-record-" + group + "-" + i).innerText = (
          record[0] + "-" + record[1] + "-" + record[2]);
      }

      if (format !== "major") {
        document.getElementById("gs-rank-team-" + group + "-" + i).innerText = team;
        document.getElementById("gs-rank-im-" + group + "-" + i).src = "image/"+team+".png";
        for (let j = 0; j < team_count; j++) {
          let prob_html = document.getElementById("group-" + group + "-team-" + i + "-rank" + j + "-prob");
          prob_html.innerText = format_prob(team_data["probs"][j], sim_data["n_samples"]);
          prob_html.style.backgroundColor = color_prob(team_data["probs"][j]);
        }
      }

      if (format === "ti")
        render_team_probs_ti(team_data, i, group, sim_data["n_samples"]);
      else if (format === "dpc")
        render_team_probs_dpc(team_data, i, group, wildcard_slots, sim_data["n_samples"]);
      else if (format === "major")
        render_team_probs_major(team_data, i, group, sim_data["n_samples"]);
    }
  }
}

function render_tiebreak_probs(format, sim_data, wildcard_slots) {
  if (format === "ti") {
    var boundaries = [[3, 7], [3, 7]];
    var groups = ['a', 'b'];
  }
  else if (format === "dpc") {
    var boundaries = [[0, 1, 2, 5], [1, 5]];
    boundaries[0][2] = wildcard_slots + 1;
    var groups = ['upper', 'lower'];
  }
  for (let group_i = 0; group_i < 2; group_i++) {
    let group = groups[group_i]
    for (let type_i = 0; type_i < boundaries[group_i].length; type_i++) {
      boundary = boundaries[group_i][type_i];
      for (let i = 0; i < 4; i++) {
        let prob_html = document.getElementById(i + "-way-tie-" + boundary + "-prob-" + group);
        let prob = sim_data["probs"]["tiebreak"][group][boundary][i];
        prob_html.innerText = format_prob(prob, sim_data["n_samples"]);
        prob_html.style.backgroundColor = color_prob(prob);
      }
      let prob_html = document.getElementById("tie-" + boundary + "-prob-" + group);
      let overall_prob = sim_data["probs"]["tiebreak"][group][boundary].reduce((a, b) => a + b, 0);
      prob_html.innerText = format_prob(overall_prob, sim_data["n_samples"]);
      prob_html.style.backgroundColor = color_prob(overall_prob);
    }
  }
}

function render_final_rank_probs(format, sim_data) {
  if (format === "ti")
    var rank_groups = 9;
  else if (format === "major")
    var rank_groups = 13;
  for (let team_i = 0; team_i < 18; team_i++) {
    let team = sim_data["probs"]["final_rank"][team_i]["team"];
    document.getElementById("final-rank-team-" + team_i).innerText = team;
    document.getElementById("final-rank-im-" + team_i).src = "image/"+team+".png";
    for (let rank_i = 0; rank_i < rank_groups; rank_i++) {
      let prob_html = document.getElementById("team-" + team_i + "-final-rank-" + rank_i + "-prob");
      let prob = sim_data["probs"]["final_rank"][team_i]["probs"][rank_i];
      prob_html.innerText = format_prob(prob, sim_data["n_samples"]);
      prob_html.style.backgroundColor = color_prob(prob);
    }
  }
}

function render_record_rank_probs(format, sim_data) {
  if (format === "ti") {
    var groups = ['a', 'b'];
    var team_count = 9;
    var possible_records = 17;
  }
  if (format === "dpc") {
    var groups = ['upper', 'lower'];
    var team_count = 8;
    var possible_records = 8;
  }
  for (let group_i = 0; group_i < 2; group_i++) {
    let group = groups[group_i]
    for (let team_i = 0; team_i < team_count; team_i++) {
      let team = sim_data["probs"]["record"][group][team_i]["team"];
      let record = "";
      if (format === "ti") {
        record = "(" + (sim_data["records"][team][0]*2 + sim_data["records"][team][1])
                 + "-" + (sim_data["records"][team][2]*2 + sim_data["records"][team][1]) + ")";
      }
      else if (format === "dpc") {
        record = "(" + sim_data["records"][team][0] + "-" + sim_data["records"][team][1] + ")";
      }
      document.getElementById("rank-record-im-" + group + "-" + team_i).src = "image/"+team+".png";
      document.getElementById("rank-record-wl-" + group + "-" + team_i).innerText = record;

      for (let record_i = 0; record_i < possible_records; record_i++) {
        let overall_prob_html = document.getElementById(group + "-team-" + team_i + "-record-" + record_i + "-prob");
        let overall_prob = sim_data["probs"]["record"][group][team_i]["record_probs"][record_i];
        overall_prob_html.innerText = format_prob(overall_prob, sim_data["n_samples"]);
        overall_prob_html.style.backgroundColor = color_prob(overall_prob, "green");

        for (let rank_i = 0; rank_i < team_count; rank_i++) {
          let prob_html = document.getElementById(group + "-team-" + team_i + "-record-" + record_i + "-rank-" + rank_i + "-prob");
          let prob = sim_data["probs"]["record"][group][team_i]["point_rank_probs"][record_i][rank_i];
          prob_html.innerText = format_prob(prob, sim_data["n_samples"]);
          prob_html.style.backgroundColor = color_prob(prob);
        }
      }
    }
  }
}

function render_ti_match(group, day, match_index, match, n_samples) {
  let team1 = match["teams"][0];
  let team2 = match["teams"][1];
  document.getElementById("match-" + group + "-" + day + "-" + match_index + "-im-1").src = "image/"+team1+".png";
  document.getElementById("match-" + group + "-" + day + "-" + match_index + "-im-2").src = "image/"+team2+".png";
  let team1_html = document.getElementById("match-" + group + "-" + day + "-" + match_index + "-team-1");
  let team2_html = document.getElementById("match-" + group + "-" + day + "-" + match_index + "-team-2");
  team1_html.innerText = team1;
  team2_html.innerText = team2;
  if (match["result"] == 2) {
    team1_html.innerHTML += " <b>(2)</b>";
    team2_html.innerHTML += " <b>(0)</b>";
  }
  if (match["result"] == 1) {
    team1_html.innerHTML += " <b>(1)</b>";
    team2_html.innerHTML += " <b>(1)</b>";
  }
  if (match["result"] == 0) {
    team1_html.innerHTML += " <b>(0)</b>";
    team2_html.innerHTML += " <b>(2)</b>";
  }

  if (match["result"] == 2) {
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-1").style.backgroundColor = "#ccffcc";
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-2").style.backgroundColor = "#ffcccc";
  }
  else if (match["result"] == 1) {
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-1").style.backgroundColor = "#fafacc";
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-2").style.backgroundColor = "#fafacc";
  }
  else if (match["result"] == 0) {
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-1").style.backgroundColor = "#ffcccc";
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-2").style.backgroundColor = "#ccffcc";
  }
  else {
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-1").style.backgroundColor = "#f8f9fa";
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-2").style.backgroundColor = "#f8f9fa";
  }

  for (let prob_i = 0; prob_i < 3; prob_i++) {
    let prob_html = document.getElementById("match-" + group + "-" + day + "-" + match_index + "-prob-" + prob_i);
    prob_html.innerText = format_prob(match["probs"][prob_i], n_samples, 0);
    prob_html.style.backgroundColor = color_prob(match["probs"][prob_i]);
  }
}

function render_dpc_match(group, day, match_index, match, n_samples) {
  let team1 = match["teams"][0];
  let team2 = match["teams"][1];
  document.getElementById("match-" + group + "-" + day + "-" + match_index + "-im-1").src = "image/"+team1+".png";
  document.getElementById("match-" + group + "-" + day + "-" + match_index + "-im-2").src = "image/"+team2+".png";
  let team1_html = document.getElementById("match-" + group + "-" + day + "-" + match_index + "-team-1");
  let team2_html = document.getElementById("match-" + group + "-" + day + "-" + match_index + "-team-2");
  team1_html.innerText = team1;
  team2_html.innerText = team2;
  if (match["result"].length > 0) {
    if ([0,2,1,"W","-"].includes(match["result"][0]))
      team1_html.innerHTML += " <b>(" + match["result"][0] + ")</b>";
    if ([0,2,1,"W","-"].includes(match["result"][1]))
      team2_html.innerHTML += " <b>(" + match["result"][1] + ")</b>";
  }

  if (match["result"].length == 0) {
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-1").style.backgroundColor = "#f8f9fa";
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-2").style.backgroundColor = "#f8f9fa";
  }
  if (match["result"][0] > match["result"][1]) {
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-1").style.backgroundColor = "#ccffcc";
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-2").style.backgroundColor = "#ffcccc";
  }
  else if (match["result"][0] < match["result"][1]) {
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-1").style.backgroundColor = "#ffcccc";
    document.getElementById("match-" + group + "-" + day + "-" + match_index + "-2").style.backgroundColor = "#ccffcc";
  }

  for (let prob_i = 0; prob_i < 2; prob_i++) {
    let prob_html = document.getElementById("match-" + group + "-" + day + "-" + match_index + "-prob-" + prob_i);
    prob_html.innerText = format_prob(match["probs"][prob_i], n_samples, 0);
    prob_html.style.backgroundColor = color_prob(match["probs"][prob_i]);
  }
}

function render_match_probs(format, sim_data) {
  if (format === "ti") {
    var groups = ['a', 'b'];
    var days = 4;
  }
  else if (format === "dpc") {
    var groups = ['upper', 'lower'];
    var days = 6;
  }
  for (let group_i = 0; group_i < 2; group_i++) {
    let group = groups[group_i]
    for (let day = 0; day < days; day++) {
      let matches = sim_data["probs"]["matches"][group][day];
      for (let match_i = 0; match_i < matches.length; match_i++) {
        let match = matches[match_i];
        if (format === "ti")
          render_ti_match(group, day, match_i, match, sim_data["n_samples"]);
        else if (format === "dpc")
          render_dpc_match(group, day, match_i, match, sim_data["n_samples"]);
      }
    }
  }
}

function render_metadata(sim_data) {
  document.querySelectorAll(".timestamp").forEach(function(element) {
    element.innerText = sim_data["timestamp"]; });

  document.querySelectorAll(".n_samples").forEach(function(element) {
    element.innerText = sim_data["n_samples"]; });
  document.querySelectorAll(".n_samples100").forEach(function(element) {
    element.innerText = 100/sim_data["n_samples"]; });
  document.querySelectorAll(".n_samples1000").forEach(function(element) {
    element.innerText = 1000/sim_data["n_samples"]; });

  document.querySelectorAll(".model-version").forEach(function(element) {
    element.innerText = sim_data["model_version"]; });
}

function render_data_ti(path) {
  fetch(path)
    .then(function(res) { return res.json(); })
    .then(function(sim_data) {
      render_group_rank_probs("ti", sim_data);
      render_tiebreak_probs("ti", sim_data);
      render_record_rank_probs("ti", sim_data);
      render_match_probs("ti", sim_data);
      render_final_rank_probs("ti", sim_data);
      render_metadata(sim_data);
    });
}

function render_data_dpc(path, wildcard_slots) {
  fetch(path)
    .then(function(res) { return res.json(); })
    .then(function(sim_data) {
      render_group_rank_probs("dpc", sim_data, wildcard_slots);
      render_match_probs("dpc", sim_data);
      render_record_rank_probs("dpc", sim_data);
      render_tiebreak_probs("dpc", sim_data, wildcard_slots);
      render_metadata(sim_data);
    });
}

function render_data_major(path) {
  fetch(path)
    .then(function(res) { return res.json(); })
    .then(function(sim_data) {
      render_group_rank_probs("major", sim_data);
      render_final_rank_probs("major", sim_data);
      render_metadata(sim_data);
    });
}
