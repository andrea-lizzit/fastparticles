using LinearAlgebra, Distributions, Random, StatsBase, Plots

function diagonal_analysis(mat::SymTridiagonal)
	Ωv = eigvals(mat)
	return Ωv
end

function ω_statistic(ω0, Δ, N::Int, t::Real)
	ωv = ω0 .+ rand(Normal(0, Δ), N)
	ham = SymTridiagonal(ωv, fill(t, N-1))
	Ωv = diagonal_analysis(ham)
	return ωv, Ωv
end

# plot a histogram of ω and a histogram of Ωv
function plot_histograms(ωv, Ωv)
	# plot histogram of ωv
	ωv_hist = fit(Histogram, ωv, nbins=50)
	p1 = plot(ωv_hist, label="ωv", xlabel="ω", ylabel="counts", legend=:topleft)
	Ωv_hist = fit(Histogram, Ωv, nbins=50)
	p2 = plot(Ωv_hist, label="Ωv", xlabel="Ω", ylabel="counts", legend=:topleft)
	return plot(p1, p2, layout=(2, 1), link=:x)
end


function plot_statistic(ω0, Δ, N::Int, t::Real)
	ωv, Ωv = ω_statistic(ω0, Δ, N, t)
	p = plot_histograms(ωv, Ωv)
	# plot!(p, title="ω0=$ω0, Δ=$Δ, N=$N, t=$t")
	title!(p, "ω0 = $ω0, Δ = $Δ, N = $N, t = $t")
	return p
end

plot_statistic(1.0, 0.1, 100, 0.3)