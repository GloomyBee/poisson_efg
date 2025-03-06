using LinearAlgebra
using SparseArrays
using Plots

# 设置 GR 后端
gr()

# 1. 定义节点和参数
#=
NP = 21                       # 节点数
nodes = collect(0.0:0.05:1.0)  # 均匀分布节点，步长 0.1，11 个点
s = 0.1                       # 影响域大小，2 倍节点间距 (0.1 * 2)
n_gauss = 2                   # 高斯积分点数
gauss_points = [-sqrt(1/3), sqrt(1/3)] # 高斯积分点
gauss_weights = [1.0, 1.0]    # 高斯积分权重
=#

# 优化后的参数

NP = 41                       # 增加节点数
nodes = collect(0.0:0.025:1.0) # 步长减小到 0.05
s = 0.05                     # 调整支撑范围
n_gauss = 3                   # 增加高斯积分点
gauss_points = [-sqrt(3/5), 0, sqrt(3/5)]  # 3 点高斯积分点
gauss_weights = [5/9, 8/9, 5/9]            # 3 点高斯积分权重


# 2. 三次样条核函数
function cubic_spline(r)
    if r <= 0.5
        return 2/3 - 4*r^2 + 4*r^3
    elseif r <= 1.0
        return 4/3 - 4*r + 4*r^2 - 4/3*r^3
    else
        return 0.0
    end
end

function cubic_spline_deriv(r)
    if r <= 0.5
        return -8*r + 12*r^2
    elseif r <= 1.0
        return -4 + 8*r - 4*r^2
    else
        return 0.0
    end
end

# 3. MLS 形函数及其导数
function shape_function(x, nodes, s)
    n = length(nodes)
    A = zeros(2, 2)
    A_deriv = zeros(2, 2)  # 新增 A 的导数
    B = zeros(2, n)       # 存储 B 矩阵
    phi = zeros(n)
    phi_deriv = zeros(n)
    Ψ = zeros(n)
    Ψ_deriv = zeros(n)

    # 第一遍循环：计算 A, A_deriv, B
    for i in 1:n
        r = abs(x - nodes[i]) / s
        phi[i] = cubic_spline(r)
        phi_deriv[i] = cubic_spline_deriv(r) * sign(x - nodes[i]) / s
        p = [1.0, (nodes[i] - x) / s]  # 归一化基函数
        B[:, i] = phi[i] * p           # B_{:,i}
        A += phi[i] * p * p'
        A_deriv += phi_deriv[i] * p * p'  # A' 只考虑 phi 的导数
    end

    if abs(det(A)) < 1e-10
        A += 1e-10 * I
    end

    inv_A = inv(A)
    # 第二遍循环：计算 Ψ 和 Ψ_deriv
    for i in 1:n
        p = [1.0, (nodes[i] - x) / s]
        p_deriv = [0.0, -1.0 / s]  # p 对 x 的导数
        B_deriv = phi_deriv[i] * p  # B_{:,i} 的导数
        B_i = B[:, i]
        
        Ψ[i] = [1.0, 0.0]' * inv_A * B_i
        Ψ_deriv[i] = (p_deriv' * inv_A * B_i) + 
                     ([1.0, 0.0]' * inv_A * (B_deriv - A_deriv * inv_A * B_i))
    end

    return Ψ, Ψ_deriv
end

# 4. 组装刚度矩阵和载荷向量
function assemble_system(nodes, s)
    NP = length(nodes)  # 节点数
    K = zeros(NP, NP)   # 刚度矩阵
    f = zeros(NP)       # 载荷向量

    elements = [(nodes[i], nodes[i+1]) for i in 1:NP-1]# 一维线性单元，每个单元由两个节点组成

    for elem in elements# 遍历每个单元
        x1, x2 = elem   # 单元的两个节点
        h = x2 - x1     # 单元长度
        for i in 1:n_gauss  # 高斯积分
            xi = gauss_points[i]
            w = gauss_weights[i]
            x = (x2 + x1) / 2 + (x2 - x1) / 2 * xi  # 坐标变换，高斯积分点的位置，xi=-1时，x=x1，xi=1时，x=x2。
            Ψ, Ψ_deriv = shape_function(x, nodes, s) #调用x 处的形函数，计算所有节点的形函数值和导数值
            
            for I in 1:NP
                for J in 1:NP
                    K[I, J] += Ψ_deriv[I] * Ψ_deriv[J] * w * h / 2 # 计算刚度矩阵
                end
                f[I] += -Ψ[I] * w * h / 2  # 计算力向量
            end
        end
    end

    return K, f
end

# 5. 施加 Dirichlet 条件（拉格朗日乘子法）
function apply_boundary_conditions(K, f, nodes)
    NP = length(nodes)
    K_aug = zeros(NP + 2, NP + 2)  # 2 个 Dirichlet 条件
    f_aug = zeros(NP + 2)
    K_aug[1:NP, 1:NP] = K
    f_aug[1:NP] = f

    # u(0) = 0
    Ψ_0, _ = shape_function(0.0, nodes, s)
    for I in 1:NP
        K_aug[I, NP+1] = Ψ_0[I]
        K_aug[NP+1, I] = Ψ_0[I]
    end
    f_aug[NP+1] = 0.0

    # u(1) = 1
    Ψ_1, _ = shape_function(1.0, nodes, s)
    for I in 1:NP
        K_aug[I, NP+2] = Ψ_1[I]
        K_aug[NP+2, I] = Ψ_1[I]
    end
    f_aug[NP+2] = 1.0

    return K_aug, f_aug
end

# 6. 主函数
function solve_poisson()
    K, f = assemble_system(nodes, s)
    K_aug, f_aug = apply_boundary_conditions(K, f, nodes)
    
    sol = K_aug \ f_aug
    d = sol[1:NP]
    λ = sol[NP+1:NP+2]

    println("节点位移解：")
    for i in 1:NP
        println("x = $(nodes[i]), u = $(d[i])")
    end
    println("拉格朗日乘子 λ = $λ")

    println("\n解析解对比：")
    for i in 1:NP
        x = nodes[i]
        u_exact = 0.5 * x * (1 + x)
        println("x = $x, u_exact = $u_exact, u_computed = $(d[i]), 误差 = $(abs(u_exact - d[i]))")
    end

    # 7. 绘制结果
    x_fine = 0:0.01:1
    u_fine = [sum(shape_function(x, nodes, s)[1] .* d) for x in x_fine]
    p = plot(x_fine, u_fine, label="EFG", xlabel="x", ylabel="u(x)", title="EFG Solution vs Exact Solution")
    plot!(p, x_fine, 0.5 .* x_fine .* (1 .+ x_fine), label="Exact", linestyle=:dash)
    scatter!(p, nodes, d, label="Computed Nodes", color=:red, markersize=4)
    display(p)  # 强制显示绘图窗口

    return d
end

# 运行
d = solve_poisson()