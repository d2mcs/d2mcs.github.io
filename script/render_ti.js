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
  if (color === "upper") {
    return "rgb(" + Math.round(235 - 200*prob) + "," + Math.round(255 - 90*prob)
            + "," + Math.round(235 - 170*prob) + ")";
  }
  else if (color === "lower") {
    return "rgb(" + Math.round(255 - 40*prob) + "," + Math.round(230 - 145*prob)
            + "," + Math.round(205 - 200*prob) + ")";
  }
  else if (color === "elim") {
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

function render_group_rank_probs(sim_data) {
  var groups = ['a', 'b'];
  var prob_types = ["upper", "lower", "elim"];
  for (let group_i = 0; group_i < 2; group_i++) {
    let group = groups[group_i]
    for (let i = 0; i < 9; i++) {
      let team_data = sim_data["probs"]["group_rank"][group][i]
      let team = team_data["team"];

      document.getElementById("gs-team-" + group + "-" + i).innerText = team;
      document.getElementById("gs-im-" + group + "-" + i).src = "image/"+team+".png";
      document.getElementById("gs-rating-" + group + "-" + i).innerText = sim_data["ratings"][team];
      let record = sim_data["records"][team];
      document.getElementById("gs-record-" + group + "-" + i).innerText = (
          record[0] + "-" + record[1] + "-" + record[2]);

      document.getElementById("gs-rank-team-" + group + "-" + i).innerText = team;
      document.getElementById("gs-rank-im-" + group + "-" + i).src = "image/"+team+".png";
      for (let j = 0; j < 9; j++) {
        let prob_html = document.getElementById("group-" + group + "-team-" + i + "-rank" + j + "-prob");
        prob_html.innerText = format_prob(team_data["probs"][j], sim_data["n_samples"]);
        prob_html.style.backgroundColor = color_prob(team_data["probs"][j]);
      }

      for (let type_i = 0; type_i < 3; type_i++) {
        let prob_type = prob_types[type_i];
        let prob_html = document.getElementById("gs-" + prob_type + "-prob-" + group + "-" + i);
        let prob = 0;
        if (prob_type == "upper")
          prob = team_data["probs"].slice(0, 4).reduce((a, b) => a + b, 0);
        else if (prob_type == "lower")
          prob = team_data["probs"].slice(4, 8).reduce((a, b) => a + b, 0);
        else
          prob = team_data["probs"][8];
        prob_html.innerText = format_prob(prob, sim_data["n_samples"]);
        prob_html.style.backgroundColor = color_prob(prob, prob_type);
      }
    }
  }
}

function render_tiebreak_probs(sim_data) {
  var boundaries = [3, 7]
  var groups = ['a', 'b'];
  for (let group_i = 0; group_i < 2; group_i++) {
    let group = groups[group_i]
    for (let type_i = 0; type_i < 2; type_i++) {
      boundary = boundaries[type_i]
      for (let i = 0; i < 4; i++) {
        let prob_html = document.getElementById(i + "-way-tie-" + boundary + "-prob-" + group);
        let prob = sim_data["probs"]["tiebreak"][group][boundary][i]
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

function render_final_rank_probs(sim_data) {
  for (let team_i = 0; team_i < 18; team_i++) {
    let team = sim_data["probs"]["final_rank"][team_i]["team"];
    document.getElementById("final-rank-team-" + team_i).innerText = team;
    document.getElementById("final-rank-im-" + team_i).src = "image/"+team+".png";
    for (let rank_i = 0; rank_i < 9; rank_i++) {
      let prob_html = document.getElementById("team-" + team_i + "-final-rank-" + rank_i + "-prob");
      let prob = sim_data["probs"]["final_rank"][team_i]["probs"][rank_i];
      prob_html.innerText = format_prob(prob, sim_data["n_samples"]);
      prob_html.style.backgroundColor = color_prob(prob);
    }
  }
}

function render_record_rank_probs(sim_data) {
  var groups = ['a', 'b'];
  for (let group_i = 0; group_i < 2; group_i++) {
    let group = groups[group_i]
    for (let team_i = 0; team_i < 9; team_i++) {
      let team = sim_data["probs"]["record"][group][team_i]["team"];
      let record = "(" + (sim_data["records"][team][0]*2 + sim_data["records"][team][1])
                   + "-" + (sim_data["records"][team][2]*2 + sim_data["records"][team][1]) + ")";
      document.getElementById("rank-record-im-" + group + "-" + team_i).src = "image/"+team+".png";
      document.getElementById("rank-record-wl-" + group + "-" + team_i).innerText = record;

      for (let record_i = 0; record_i < 17; record_i++) {
        let overall_prob_html = document.getElementById(group + "-team-" + team_i + "-record-" + record_i + "-prob");
        let overall_prob = sim_data["probs"]["record"][group][team_i]["record_probs"][record_i];
        overall_prob_html.innerText = format_prob(overall_prob, sim_data["n_samples"]);
        overall_prob_html.style.backgroundColor = color_prob(overall_prob, "green");

        for (let rank_i = 0; rank_i < 9; rank_i++) {
          let prob_html = document.getElementById(group + "-team-" + team_i + "-record-" + record_i + "-rank-" + rank_i + "-prob");
          let prob = sim_data["probs"]["record"][group][team_i]["point_rank_probs"][record_i][rank_i];
          prob_html.innerText = format_prob(prob, sim_data["n_samples"]);
          prob_html.style.backgroundColor = color_prob(prob);
        }
      }
    }
  }
}

function render_match_probs(sim_data) {
  var groups = ['a', 'b'];
  for (let group_i = 0; group_i < 2; group_i++) {
    let group = groups[group_i]
    for (let day = 0; day < 4; day++) {
      let matches = sim_data["probs"]["matches"][group][day];
      for (let match_i = 0; match_i < matches.length; match_i++) {
        let match = matches[match_i];
        let team1 = match["teams"][0];
        let team2 = match["teams"][1];
        document.getElementById("match-" + group + "-" + day + "-" + match_i + "-im-1").src = "image/"+team1+".png";
        document.getElementById("match-" + group + "-" + day + "-" + match_i + "-im-2").src = "image/"+team2+".png";
        let team1_html = document.getElementById("match-" + group + "-" + day + "-" + match_i + "-team-1");
        let team2_html = document.getElementById("match-" + group + "-" + day + "-" + match_i + "-team-2");
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
          document.getElementById("match-" + group + "-" + day + "-" + match_i + "-1").style.backgroundColor = "#ccffcc";
          document.getElementById("match-" + group + "-" + day + "-" + match_i + "-2").style.backgroundColor = "#ffcccc";
        }
        else if (match["result"] == 1) {
          document.getElementById("match-" + group + "-" + day + "-" + match_i + "-1").style.backgroundColor = "#fafacc";
          document.getElementById("match-" + group + "-" + day + "-" + match_i + "-2").style.backgroundColor = "#fafacc";
        }
        else if (match["result"] == 0) {
          document.getElementById("match-" + group + "-" + day + "-" + match_i + "-1").style.backgroundColor = "#ffcccc";
          document.getElementById("match-" + group + "-" + day + "-" + match_i + "-2").style.backgroundColor = "#ccffcc";
        }
        else {
          document.getElementById("match-" + group + "-" + day + "-" + match_i + "-1").style.backgroundColor = "#f8f9fa";
          document.getElementById("match-" + group + "-" + day + "-" + match_i + "-2").style.backgroundColor = "#f8f9fa";
        }

        for (let prob_i = 0; prob_i < 3; prob_i++) {
          let prob_html = document.getElementById("match-" + group + "-" + day + "-" + match_i + "-prob-" + prob_i);
          prob_html.innerText = format_prob(match["probs"][prob_i], sim_data["n_samples"], 0);
          prob_html.style.backgroundColor = color_prob(match["probs"][prob_i]);
        }
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

function render_data(path) {
  fetch(path)
    .then(function(res) { return res.json(); })
    .then(function(sim_data) {
      render_group_rank_probs(sim_data);
      render_tiebreak_probs(sim_data);
      render_final_rank_probs(sim_data);
      render_record_rank_probs(sim_data);
      render_match_probs(sim_data);
      render_metadata(sim_data);
    });
}
